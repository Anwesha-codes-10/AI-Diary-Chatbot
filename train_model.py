import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import time

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
train_dataset = dataset["train_sft"]
eval_dataset = dataset["test_sft"]

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False,trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model.gradient_checkpointing_enable()
peft_model = get_peft_model(model, lora_config)

def tokenize_function(example):
    return tokenizer(example["prompt"], truncation=True,padding="max_length", max_length=256)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=f"./phi2_qlora_output_{int(time.time())}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    save_steps=25,
    max_steps=1000,
    gradient_checkpointing=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()


trainer.save_model("./fine_tuned_phi2_qlora")
tokenizer.save_pretrained("./fine_tuned_phi2_qlora")

