
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from transformers import GenerationConfig


model_path = "./llama-160m"


########################################################################

def training_args():
    params = torch.load(model_path + "/training_args.bin")
    params = dict(params.__dict__)
    for k, v in params.items():
        print(f"{k}: {v}")
#training_args()


gen_config = GenerationConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path) # = LlamaTokenizer

model = AutoModelForCausalLM.from_pretrained(model_path)

total_bytes = sum(p.numel() for p in model.parameters())
print(f"Model.params={total_bytes / (1000 ** 2):.2f} MB")

########################################################################

text = "The capital of France is"

inputs = tokenizer(text, return_tensors="pt")


outputs = model.generate(
    **inputs,
    generation_config=gen_config,
    max_new_tokens=24,
    do_sample=False,
    repetition_penalty=1.1,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_text = str(generated_text).replace('\n', ' ')

print("Generated:", generated_text)
