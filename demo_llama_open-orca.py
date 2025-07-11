
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


max_length=2048

model_path = "./llama-160m"


def training_args():
    params = torch.load(model_path + "/training_args.bin")
    params = dict(params.__dict__)
    for k, v in params.items():
        print(f"{k}: {v}")
#training_args()

tokenizer = AutoTokenizer.from_pretrained(model_path) # = LlamaTokenizer
print(f"bos={tokenizer.bos_token_id}, pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")

#tokenizer.pad_token_id = tokenizer.eos_token_id


# --- 2. Create config for LLaMA model
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,            # 768
    intermediate_size=2048,     # 3072
    num_attention_heads=8,      # 12
    num_hidden_layers=4,
    max_position_embeddings=2048,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# --- 3. Initialization from zero
model = LlamaForCausalLM(config)


# --- 4. Load dataset (streaming-mode, to avoid VRAM overload )
dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=False)

print(f"Size: {len(dataset)}")


# --- 2. Preprocessing ---
def preprocess(example):
    return {
        "text": f"<|system|>\n{example['system_prompt']}\n<|user|>\n{example['question']}\n<|assistant|>\n{example['response']}"
    }

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

exit(0)


# --- 4. Tokenization ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# --- 5. DataLoader ---
def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_loader = DataLoader(
    tokenized_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)
