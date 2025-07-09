
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

exit(0)

# --- 4. Загружаем датасеты C4 (стриминг, чтобы не грузить весь объем в память)
# \%User%\.cache\huggingface\hub
c4_train = load_dataset("allenai/c4", "en", split="train", streaming=False)

# Фильтруешь весь train по subset
c4_realnews = c4_train.filter(lambda x: x["subset"] == "realnewslike")
c4_en = c4_train.filter(lambda x: x["subset"] == "en")

# --- 5. Объединим два датасета в один генератор
def dataset_generator():
    for example in c4_en:
        if example["text"]:
            yield {"text": example["text"]}
    for example in c4_realnews:
        if example["text"]:
            yield {"text": example["text"]}

dataset = list(dataset_generator())  # Для небольшого объёма. Для большого лучше писать кастомный DataLoader


dataset = Dataset.from_list(dataset)

# --- 6. Токенизация
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- 7. Группировка токенов в блоки для обучения (токенизация по длине блоков)
block_size = 1024

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    input_ids = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": input_ids, "labels": input_ids}

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# --- 8. Настраиваем DataCollator (без MLM, т.к. это causal LM)
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
    train_dataset=lm_dataset,
    data_collator=data_collator
)

# --- 11. Запускаем обучение
#trainer.train()
