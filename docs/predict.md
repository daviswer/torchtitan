# Get predictions for next word

## Setup the Model
```
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
model_config = models_config["llama3"]["3B"]
model_config.vocab_size=128256
model_cls = model_name_to_cls["llama3"]

#from torchtitan.models.llama.model import ModelArgs
#model_config=ModelArgs(dim=3072, n_layers=24, n_heads=32, n_kv_heads=8, vocab_size=128256, multiple_of=1024, ffn_dim_multiplier=1.0, norm_eps=1e-05, rope_theta=500000, max_batch_size=32, max_seq_len=4096, depth_init=True, norm_type='rmsnorm')

import torch
# If you have some arbitrary code which performs some tensor construction without explicitly specifying a device, you can override it to instead construct on meta device by using the torch.device() context manager
# This is especially helpful NN module construction, where you often are not able to explicitly pass in a device for initialization:
with torch.device("meta"): model = model_cls.from_model_args(model_config)
```

## Load the checkpoint using one of the mechanisms below:

### Load the distributed checkpoint
```
# NN modules have a convenience method torch.nn.Module.to_empty() that allow you to the module to another device, leaving all parameters uninitialized. You are expected to explicitly reinitialize the parameters manually
model.to_empty(device="cuda")
#model.to_empty(device="cpu")
state_dict = {"model": model.state_dict(),}
# since no progress group is initialized, DCP will disable any collectives.
import torch.distributed.checkpoint as DCP
DCP.load(state_dict=state_dict,checkpoint_id="../checkpoint/step-100000/",)
model.load_state_dict(state_dict["model"])
```

### Load from sharded checkpoints that are converted to a single file (if already available as shown in References)
```
checkpoint_file="../checkpoint.pt"
checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint["model"])
```

## Load the tokenizer
```
from torchtitan.datasets import build_tokenizer
tokenizer = build_tokenizer(model_name_to_tokenizer["llama3"], "/proj/data-eng/tokenizers-llama3/tokenizer.model") # tiktoken
```

## Steps to predict the next word
```
input = torch.tensor([tokenizer.encode('Capital of India is',bos='<|begin_of_text|>',eos=None)])
#input = torch.tensor([tokenizer.encode('I enjoy walking in the',bos=128000,eos=None)])
input=input.to("cuda")
with torch.no_grad(): preds=model(input)

# 1. get the predictions for the last logits
next_token_logits = preds[0, -1, :]
print(next_token_logits.shape)
print(next_token_logits)

# 2. step to convert the logits to probabilities
next_token_probs = torch.softmax(next_token_logits, -1)

# 3. step to get the top 10
topk_next_tokens= torch.topk(next_token_probs, 10)

print(*[(tokenizer.decode(list([idx])), prob) for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)], sep="\n")
```

## Create a class LMHeadModel for predictions
```
class LMHeadModel:
    def __init__(self, model, tokenizer):
        # Initialize the model and the tokenizer.
        self.model = model
        self.tokenizer = tokenizer   
    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = torch.tensor([self.tokenizer.encode(sentence,bos=128000,eos=None)]).to("cuda")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    def get_next_word_probabilities(self, sentence, top_k=500):
        # Get the model predictions for the sentence.
        inputs = torch.tensor([self.tokenizer.encode(sentence,bos=128000,eos=None)]).to("cuda")
        predictions = self.model(inputs)
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]
        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(next_token_candidates_tensor, top_k).indices.tolist()
        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(next_token_candidates_tensor, dim=-1)
        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = all_candidates_probabilities[topk_candidates_indexes].tolist()
        # Decode the top k candidates back to words.
        topk_candidates_tokens = [self.tokenizer.decode(list([idx])).strip() for idx in topk_candidates_indexes]
        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))

llm=LMHeadModel(model,tokenizer)
llm.get_predictions("Capital of India is New")
llm.get_next_word_probabilities("Capital of India is New",top_k=5)
```

## References
- Convert Sharded Checkpoints to a Single File https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md
    ```
    python -m torch.distributed.checkpoint.format_utils dcp_to_torch torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
    ```
- DCP Checkpoint https://www.youtube.com/watch?v=ldBmHNva_Fw 
- Distributed Checkpoint Recipe https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html 


