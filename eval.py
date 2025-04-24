import argparse
import os
from typing import List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
import lm_eval
from lm_eval.utils import make_table
from lm_eval.api.instance import Instance  # type: ignore
from lm_eval.api.model import LM  # type: ignore
from lm_eval.api.registry import register_model  # type: ignore
from torch import nn
from torch import distributed as dist
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict

from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

def has_package(name):
    """
    Checks if a package is installed and available.
    """
    try:
        __import__(name)
    except ImportError:
        return False
    else:
        return True


# TOKENIZERS

char_tokenizer = "char_tokenizer"
gpt_neox_20b = "EleutherAI/gpt-neox-20b"
gpt_neox_125m = "EleutherAI/gpt-neox-125M"


_has_hf = has_package("transformers")
_has_sp = has_package("sentencepiece")


class BaseTokenizer:
    """
    A simplistic tokenizer interface duck-type compatible with HuggingFace
    tokenizers. An implementation of this interface could be used in fm in
    cases where we'd like to write tests that don't depend on HF.
    """

    def __init__(self, bos_id: int, eos_id: int):
        """
        bos_id: the ID representing the beginning-of-sentence token
        eos_id: the ID representing the end-of-sentence token
        """
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id

    def tokenize(self, text: str):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        """
        for all tokenizers, a str parameter will be interpreted as a single token,
        and its output will be a single integer that represents the id.
        """
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]):
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """
    This is essentially the tokenizer used by minGPT. Every character
    is a token, tokenized as ord(c). Vocab size is 256.
    """

    def __init__(self):
        # 2, 3 from ascii tables are "start of text" and "end of text"
        super().__init__(2, 3)

    def tokenize(self, text: str):
        return list(text)

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        return [chr(i) for i in ids]

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        if isinstance(tokens, str):
            # returning a single integer to be compatible with other tokenizers
            if len(tokens) != 1:
                raise RuntimeError(
                    "Only single character str tokens can be converted using the CharTokenizer."
                )
            token_id = ord(tokens)
            return token_id if token_id < 256 else 0
        return [ord(t) if len(t) == 1 and ord(t) < 256 else 0 for t in tokens]

    def convert_tokens_to_string(self, tokens: list[str]):
        return "".join(tokens)

    def vocab_size(self):
        return 256


class _SentencePieceTokenizer(BaseTokenizer):
    """
    An adapter for a sentencepiece tokenizer.
    """

    def __init__(self, path: str):
        from sentencepiece import SentencePieceProcessor  # type: ignore

        self.sp_model = SentencePieceProcessor(model_file=path)
        super().__init__(self.sp_model.bos_id(), self.sp_model.eos_id())

    def tokenize(self, text: str):
        return self.sp_model.encode_as_pieces(text)

    def convert_ids_to_tokens(self, ids: Union[List[int], torch.LongTensor]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp_model.id_to_piece(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        return self.sp_model.piece_to_id(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
        return self.sp_model.decode(tokens)

    def vocab_size(self):
        return self.sp_model.vocab_size()


class _HFTokenizer(BaseTokenizer):
    """
    An adapter over a HuggingFace tokenizer.
    """

    def __init__(self, name: str):
        from transformers import AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        super().__init__(self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def get_tokenizer(name: str, style: Optional[str] = None) -> BaseTokenizer:
    """
    Hack to get an instance of a tokenizer by name or path.

    Args:

    style: 'hf', 'sentencepiece', or 'fms'. If not specified, attempt to derive
            the type based on the name.
    """
    if name == "char_tokenizer" and (style is None or style == "fms"):
        return CharTokenizer()

    # SentencePiece saves models as .model files.
    # It would be better to identify the type of the file accurately, e.g. using protobuf:
    # https://github.com/google/sentencepiece/issues/121
    if style == "sentencepiece" or (
        style is None
        and len(name) >= len(".model")
        and name[-len(".model") :] == ".model"
    ):
        name = os.path.expanduser(name)
        if not os.path.exists(name):
            raise RuntimeError(f"Could not find SentencePiece model at '{name}'")
        if not _has_sp:
            raise RuntimeError(
                f"'{name}' appears to be a sentencepiece tokenizer but sentencepiece is not installed"
            )
        return _SentencePieceTokenizer(name)
    if not _has_hf:
        raise RuntimeError(
            f"Could not find tokenizer '{name}' and HuggingFace transformers is not installed"
        )
    if style is None or style == "hf":
        return _HFTokenizer(name)
    if style is None:
        raise RuntimeError(f"Could not find a tokenzier {name}")
    else:
        raise RuntimeError(f"Could not find a {style} tokenizer with name {name}")


# EVAL

@register_model("fms")
class FMSEvalHarnessLM(LM):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        device="cpu",
        rank=0,
        world_size=1,
    ):
        self.wrapped_model = model
        self.tokenizer = tokenizer
        self._rank = rank
        self._world_size = world_size
        self.device = device

        # workaround for https://github.com/EleutherAI/lm-evaluation-harness/issues/1333
        # until the fix is in a release
        def generic_object():
            return None

        self.model = generic_object
        self.model.config = generic_object  # type: ignore
        self.model.config._name_or_path = "FMSEvalHarnessLM"  # type: ignore

    def loglikelihood_one(self, context: str, continuation: str) -> Tuple[float, bool]:
        context_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context)
        )
        if not len(context_ids):
            context_ids = [self.tokenizer.bos_token_id]

        continuation_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(continuation)
        )
        input_ids = context_ids + continuation_ids[:-1]
        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        logits = F.log_softmax(self.wrapped_model(input_ids)[0], -1)
        continuation_probs = logits[len(context_ids) - 1 :]
        loglikelihood = torch.gather(
            continuation_probs, 1, torch.tensor(continuation_ids).unsqueeze(1)
        ).squeeze()
        predicted = torch.argmax(continuation_probs, -1).tolist()
        greedy = predicted == continuation_ids
        return loglikelihood.sum().cpu().item(), greedy

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        result = []
        for request in requests:
            context, continuation = request.args
            result.append(self.loglikelihood_one(context, continuation))
        return result

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("not implemented yet")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("not implemented yet")


# SCRIPT

parser = argparse.ArgumentParser(description="Script to evaluate a causal model")
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
# parser.add_argument(
#     "--compile",
#     action="store_true",
#     help="Use torch.compile (slow for first inference pass)",
# )
# parser.add_argument(
#     "--compile_mode",
#     type=str,
#     help="Mode for compilation",
#     default="default",
#     choices=["default", "reduce-overhead"],
# )
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
# parser.add_argument(
#     "--distributed",
#     action="store_true",
#     help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
# )
parser.add_argument("--tasks", type=str, help="Task names to pass to lm_eval")
parser.add_argument(
    "--num_fewshot",
    type=int,
    default=None,
    help="Number of examples in few-shot context",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

# if args.distributed:
#     dist.init_process_group()
#     # Fix until PT 2.3
#     torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
# if args.distributed:
#     distr_param = "tp"
# else:
#     if torch.cuda.device_count() > 1 and world_size == 1:
#         distr_param = "mp"
#     else:
#         distr_param = None

# model = get_model(
#     args.architecture,
#     args.variant,
#     model_path=args.model_path,
#     device_type=args.device_type,
#     source=args.model_source,
#     distributed_strategy=distr_param,
#     group=dist.group.WORLD,
# )
model_name = "llama3"
model_cls = model_name_to_cls[model_name] # TODO import
model_config = models_config[model_name][args.architecture] # TODO import
model_config.vocab_size = 128256
model_config.max_seq_len = 4096
model = model_cls.from_model_args(model_config)
model.init_weights()

# load state dict
state_dict = {"model_state": model.state_dict()}
print(state_dict["model_state"].keys())
load_state_dict(
    state_dict=state_dict, storage_reader=FileSystemReader(args.model_path), no_dist=True
)
model.load_state_dict(state_dict["model_state"])

tokenizer = get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

# if args.compile:
#     print("compiling model")
#     # Bug with kv-cache in PT2.1
#     torch._inductor.config.joint_graph_constant_folding = False
#     # compiling can make first inference pass slow
#     model = torch.compile(model, mode=args.compile_mode)


lm_obj = FMSEvalHarnessLM(model=model, tokenizer=tokenizer, device=device)

results = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=args.tasks.split(","),
    num_fewshot=args.num_fewshot,
)
print(make_table(results))
if "groups" in results:
    print(make_table(results, "groups"))
