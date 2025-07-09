
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


model_path = "./llama-68m"


def training_args():
    params = torch.load(model_path + "/training_args.bin")
    params = dict(params.__dict__)
    for k, v in params.items():
        print(f"{k}: {v}")
#training_args()

tokenizer = AutoTokenizer.from_pretrained(model_path) # = LlamaTokenizer


print(f"bos={tokenizer.bos_token_id}, pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")

# --- 2. Создаем конфигурацию модели LLaMA (пример для небольшой модели)
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,     # из токенизатора
    hidden_size=768,                     # размер эмбеддингов (параметры настраивай под железо)
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=2,
    max_position_embeddings=2048,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# --- 3. Инициализируем модель с нуля
model = LlamaForCausalLM(config)

total_bytes = sum(p.numel() for p in model.parameters())
print(f"Model.params={total_bytes / (1024 ** 2):.2f} MB")


# --- 4. Загружаем датасеты C4 (стриминг, чтобы не грузить весь объем в память)
# \%User%\.cache\huggingface\hub
dataset = load_dataset("gair-prox/FineWeb-pro", split="train", streaming=False)

max_length=2048

print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Preparing data collator...")   # mlm=False, т.к. это causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- 9. Конфиг для обучения
training_args = TrainingArguments(
    output_dir="./llama_from_scratch",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    save_steps=500,
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

# --- 10. Создаем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# --- 11. Запускаем обучение
#trainer.train()
