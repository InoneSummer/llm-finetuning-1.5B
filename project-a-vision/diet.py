from datasets import load_dataset
from PIL import Image
import os

print("🔥 초극한 다이어트 (이미지 축소 + 텍스트 2000자 절단) 시작...")

# 1. 원본 데이터셋 불러오기
ds = load_dataset("parquet", data_files="./mlx_ready_dataset/train.parquet")["train"]

def extreme_diet(example):
    # 2. 텍스트 강제 절단 (2000자 제한)
    new_msgs = []
    for msg in example["messages"]:
        text = msg["content"]
        if isinstance(text, str) and len(text) > 2000:
            text = text[:2000] + "\n"
        new_msgs.append({"role": msg["role"], "content": text})
    example["messages"] = new_msgs

    # 3. 이미지 물리적 축소 (가장 긴 변을 448로 제한)
    # 원본 이미지를 덮어씌워 메모리 폭발을 원천 차단합니다.
    img = example["images"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((448, 448), Image.Resampling.LANCZOS)
    example["images"] = img

    return example

# 4. 데이터 변환 및 저장
diet_ds = ds.map(extreme_diet)
output_dir = "./mlx_ready_dataset_extreme"
os.makedirs(output_dir, exist_ok=True)
diet_ds.to_parquet(f"{output_dir}/train.parquet")

print(f"✅ 완료! 가장 가벼운 최종 데이터셋 폴더: {output_dir}")