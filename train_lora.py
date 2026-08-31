import os
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.2-3b-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset("json", data_files="cowden_lora_fixed.jsonl", split="train")

# make a text column - no formatting_func needed
def make_text(examples):
    return {"text": [f"### Instruction:\n{ins}\n\n### Response:\n{out}" for ins, out in zip(examples["instruction"], examples["output"])]}

dataset = dataset.map(make_text, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        output_dir="cowden_lora_fixed_model",
        save_strategy="epoch",
        seed=3407,
    ),
)
trainer.train()
model.save_pretrained("cowden_lora_fixed_model")
tokenizer.save_pretrained("cowden_lora_fixed_model")
print("DONE - saved to cowden_lora_fixed_model")