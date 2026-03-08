from datasets import load_dataset
from unsloth import FastVisionModel
from PIL import Image
import torch

# 테스트 샘플 로드
print("Loading test samples...")
ds = load_dataset("ronantakizawa/webui", split="test")
ds_filtered = ds.filter(lambda x: len(x['html']) <= 4000)
samples = ds_filtered.select(range(3))  # 3개 테스트

# 모델 로드 (베이스라인 - 파인튜닝 전)
print("Loading base model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

def generate_html(model, tokenizer, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Generate the HTML code for this UI screenshot."}
            ]
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# 베이스라인 결과
print("\n=== BASELINE (before fine-tuning) ===")
baseline_results = []
for i, sample in enumerate(samples):
    print(f"\n--- Sample {i+1} ---")
    result = generate_html(model, tokenizer, sample['image'])
    baseline_results.append(result)
    print(f"Generated:\n{result[:500]}")
    print(f"\nGround Truth:\n{sample['html'][:500]}")

# 파인튜닝된 모델 로드
print("\n\nLoading fine-tuned model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "./adapters_qwen",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# 파인튜닝 결과
print("\n=== FINE-TUNED (after fine-tuning) ===")
for i, sample in enumerate(samples):
    print(f"\n--- Sample {i+1} ---")
    result = generate_html(model, tokenizer, sample['image'])
    print(f"Generated:\n{result[:500]}")
    print(f"\nGround Truth:\n{sample['html'][:500]}")

print("\nDone!")
