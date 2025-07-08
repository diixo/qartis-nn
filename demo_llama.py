import pickle
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


model_path = "./llama-68m"


params = torch.load(model_path + "/training_args.bin")
params = dict(params.__dict__)
for k, v in params.items():
    print(f"{k}: {v}")


tokenizer = LlamaTokenizer.from_pretrained(model_path)

# --- 2. Создаем конфигурацию модели LLaMA (пример для небольшой модели)
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,     # из токенизатора
    hidden_size=512,                     # размер эмбеддингов (параметры настраивай под железо)
    intermediate_size=2048,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=2048,
)

# --- 3. Инициализируем модель с нуля
model = LlamaForCausalLM(config)

# --- 4. Загружаем датасеты C4 (стриминг, чтобы не грузить весь объем в память)
c4_en = load_dataset("allenai/c4", "en", split="train", streaming=False)
c4_realnews = load_dataset("allenai/c4", "en", split="realnewslike", streaming=False)

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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_steps=10000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True,
    optim="adamw_torch",
    warmup_steps=1000,
    report_to="none"
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
