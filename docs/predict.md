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
    def get_predictions(self, sentence, eos_token: int = None):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = torch.tensor([self.tokenizer.encode(sentence,bos=128000,eos=eos_token)]).to("cuda")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    def get_next_word_probabilities(self, sentence, top_k:int = 500, temperature:float = 1.0, eos_token: int = None, ):
        if temperature <= 0: temperature = 1e-10  # Prevent division by zero or negative temperatures
        # Get the model predictions for the sentence.
        inputs = torch.tensor([self.tokenizer.encode(sentence,bos=128000,eos=eos_token)]).to("cuda")
        predictions = self.model(inputs)
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]/temperature
        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(next_token_candidates_tensor, top_k).indices.tolist()
        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(next_token_candidates_tensor, dim=-1)
        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = all_candidates_probabilities[topk_candidates_indexes].tolist()
        # Decode the top k candidates back to words.
        topk_candidates_tokens = [self.tokenizer.decode(list([idx])) for idx in topk_candidates_indexes]
        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities,topk_candidates_indexes))
    def generate(self, sentence, max_tokens_to_generate: int=100, top_k:int = 500, temperature: float = 1.0, eos_word = ".", ):
        for _ in range(max_tokens_to_generate):
            next_token_probabilities = self.get_next_word_probabilities(sentence, top_k=top_k, temperature=temperature, eos_token=None)
            # Sample the next token from the probability distribution
            next_token_index = torch.multinomial(torch.tensor([j for (_,j,_) in next_token_probabilities], dtype=torch.float), num_samples=1)[0].item()
            #print(next_token_index)
            next_word=next_token_probabilities[next_token_index]
            #print(next_word)
            sentence=sentence+next_word[0]
            if eos_word is not None and next_word[0] == eos_word: break
        return sentence
```

## Perform predictions
```
llm=LMHeadModel(model,tokenizer)
llm.get_predictions("Capital of India is New")
llm.get_next_word_probabilities("Capital of India is New",top_k=5,temperature=1.0)
llm.get_next_word_probabilities("Capital of India is New",top_k=5,temperature=0.9)
llm.get_next_word_probabilities("Capital of India is New",top_k=5,temperature=2.0)

llm=LMHeadModel(model,tokenizer)
llm.generate("Capital of India is",max_tokens_to_generate=50,top_k=5,temperature=1.0001)
llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=1.0001)
```
## Parameter: temperature T
* When T>1: Increasing T makes the distribution more uniform (flatter), increasing the randomness of the sampling. This happens because as T grows, the difference between the largest and smallest logits has less impact on the resulting probabilities, making less likely events more probable.
* When T=1: The model behaves normally, with no modification to the logits before applying the softmax function. The probabilities reflect the model’s learned distribution.
* When T<1: Decreasing T makes the distribution sharper, reducing randomness. Lower temperatures increase the disparity between the higher and lower logits, making the model’s predictions more deterministic (i.e., the highest logit value(s) dominate the probability distribution).

## Output
```
>>> llm=LMHeadModel(model,tokenizer)
>>> llm.generate("Capital of India is",max_tokens_to_generate=50,top_k=5,temperature=1.0001)
'Capital of India is New Delhi.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=1.0001)
'I enjoy walking in the park and enjoying the beauty of the trees and flowers.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=1.1)
'I enjoy walking in the forest, and especially in the springtime.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=2.1)
'I enjoy walking in the woods, but the last couple years of drought have caused a few of my plants to wilt or even die.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=2.1)
'I enjoy walking in the woods at the beginning of each week to find out about nature.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=2.1)
'I enjoy walking in the forest, especially at night and at the beginning.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=2.1)
'I enjoy walking in the park, especially the trails that surround the city of Chicago.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=2.1)
'I enjoy walking in the park at night with a friend.'
>>> llm.generate("I enjoy walking in the",max_tokens_to_generate=50,top_k=5,temperature=0)
'I enjoy walking in the woods and watching the birds.'
```

## Convert to Huggingface format
### Convert Sharded Checkpoints to a Single File
DCP saves checkpoints in a format which is inherently different then those generated using torch.save. The format_utils module in torch.distributed.checkpoint.format_utils allows conversion to the torch.save format using the mode [dcp_to_torch](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html#formats)
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch checkpoint/step-500000 checkpoint-500000.pt
mkdir single-checkpoint
cp checkpoint-500000.pt single-checkpoint/consolidated.00.pth
cp tokenizer.model single-checkpoint/.
```

### Create the single-checkpoint/params.json
```
{
"dim":3072, "n_layers":24, "n_heads":32, "n_kv_heads":8, "vocab_size":128256, "multiple_of":1024, "ffn_dim_multiplier":1.0, "norm_eps":1e-05, "rope_theta":500000, "max_batch_size":32, "max_seq_len":4096, "depth_init":true, "norm_type":"rmsnorm"
}
```

### Changes to src/transformers/models/llama/convert_llama_weights_to_hf.py
Clone the transformers repo
```
git clone https://github.com/huggingface/transformers.git
cd transformers
```

Edit the src/transformers/models/llama/convert_llama_weights_to_hf.py
```
 NUM_SHARDS = {
+    "3B": 1,
     "7B": 1,
     "8B": 1,
     "8Bf": 1,

-CONTEXT_LENGTH_FOR_VERSION = {"3.1": 131072, "3": 8192, "2": 4096, "1": 2048}
+CONTEXT_LENGTH_FOR_VERSION = {"3.1": 4096, "3": 4096, "2": 4096, "1": 2048}

def write_model(
     vocab_size=None,
     num_shards=None,
     instruct=False,
+    file_name="consolidated.00.pth",
 ):



     print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
     # Load weights
     if num_shards == 1:
         # Not sharded
         # (The sharded implementation would also work, but this is simpler.)
+        loaded = torch.load(os.path.join(input_base_path, file_name), map_location="cpu")
+        loaded=loaded.get("model")
     else:

```

###  Convert to Huggingface format
```
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../single-checkpoint --model_size 3B --output_dir ../output-checkpoint --llama_version 3
```

## Convert multiple checkpoint step folders
convert_checkpoints.sh # Run in torchtitan folder
```
#!/bin/bash
for ckp in `ls ../checkpoint`; do
  if [ ! -e ../single-checkpoint/$ckp.pt ]; then
    echo python -m torch.distributed.checkpoint.format_utils dcp_to_torch ../checkpoint/$ckp ../single-checkpoint/$ckp.pt
    python -m torch.distributed.checkpoint.format_utils dcp_to_torch ../checkpoint/$ckp ../single-checkpoint/$ckp.pt
  else
    echo ../single-checkpoint/$ckp.pt already available
  fi
done
```

convert_checkpoints_hf.sh # Run in transformers folder
```
#!/bin/bash
for ckp in `ls ../single-checkpoint/*.pt | xargs -n 1 basename -s .pt`; do
  echo $ckp
  if [ ! -e ../output-checkpoint/$ckp ]; then
    echo python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../single-checkpoint --file_name $ckp.pt --model_size 3B --output_dir ../output-checkpoint/$ckp --llama_version 3
    python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../single-checkpoint --file_name $ckp.pt --model_size 3B --output_dir ../output-checkpoint/$ckp --llama_version 3
  fi
done
```

### Run the model generator using the huggingface format
```
from transformers import pipeline
generator = pipeline('text-generation', model = '/proj/data-eng/blog/output-checkpoint',device="cuda")
generator("The capital of India is", max_length = 20, truncation=True)
```

Output
```
>>> from transformers import pipeline
>>> generator = pipeline('text-generation', model = '/proj/data-eng/blog/hfmodel/fp8_fineweb/')
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:44<00:00, 22.04s/it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Hardware accelerator e.g. GPU is available in the environment, but no `device` argument is passed to the `Pipeline` object. Model will be on CPU.
>>> generator("The capital of India is", max_length = 20, truncation=True)
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
[{'generated_text': 'The capital of India is New Delhi. The capital of India is New Delhi. The capital of India'}]
>>> generator("I enjoy walking in the", max_length = 20)
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
[{'generated_text': 'I enjoy walking in the woods and I love to watch the birds. I have a bird feeder in'}]
```

## References
- Sharded checkpoint to single file https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md
- DCP Checkpoint https://www.youtube.com/watch?v=ldBmHNva_Fw 
- Distributed Checkpoint Recipe https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html 
