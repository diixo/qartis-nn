
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


max_length=2048

#model_path = "./llama-160m"
model_path = "JackFram/llama-160m" # load from huggingface

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path) # = LlamaTokenizer

print(f"bos={tokenizer.bos_token_id}, pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- 4. Load dataset
dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=False)

print(f"Size: {len(dataset)}")


# --- 2. Preprocessing ---
def preprocess(example):
    return {
        "text": f"<|system|>\n{example['system_prompt']}\n<|user|>\n{example['question']}\n<|assistant|>\n{example['response']}"
    }

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)



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
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

#train_loader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


training_args = TrainingArguments(
    output_dir="./llama_openorca_checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.005,
    fp16=True,
    optim="adamw_torch",
    warmup_steps=0,
    report_to="none",
    eval_steps=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collate_fn,
)


trainer.train()
