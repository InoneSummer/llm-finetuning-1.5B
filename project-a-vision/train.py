from datasets import load_dataset
from unsloth import FastVisionModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

# 1. 모델 로드
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    random_state=42,
)

# 2. 데이터 로드 및 필터링
print("Loading dataset...")
ds = load_dataset("ronantakizawa/webui")
train_data = ds['train'].filter(lambda x: len(x['html']) <= 4000)
val_data = ds['validation'].filter(lambda x: len(x['html']) <= 4000)
train_data = train_data.select(range(min(2000, len(train_data))))
val_data = val_data.select(range(min(200, len(val_data))))
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# 3. 데이터 포맷 변환
def format_sample(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": "Generate the HTML code for this UI screenshot."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["html"]}
                ]
            }
        ]
    }

train_data = [format_sample(s) for s in train_data]
val_data = [format_sample(s) for s in val_data]

# 4. 학습
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_data,
    eval_dataset=val_data,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        output_dir="./adapters_qwen",
        optim="adamw_8bit",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

print("Starting training...")
trainer.train()
print("Done!")
model.save_pretrained("./adapters_qwen")
tokenizer.save_pretrained("./adapters_qwen")
