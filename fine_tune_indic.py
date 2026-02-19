# Line 91-120: Fine-tuning setup
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load Indic dataset (e.g., Hindi QA from IndicGLUE)
dataset = load_dataset("ai4bharat/IndicGLUE", "amrithabhashini")  # Or "indicqa" for QA
dataset = dataset['train'].train_test_split(test_size=0.1)

# Tokenizer & model (base Llama)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# LoRA config for efficiency
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# Tokenize function
def tokenize_func(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_func, batched=True)

# Training args (tune on EC2 for speed)
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama_indic",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True  # If GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

trainer.train()

# Save & convert to GGUF for llama.cpp
model.save_pretrained("./fine_tuned_llama_indic")
# Use unsloth or llama.cpp tools to convert to GGUF (manual step)
