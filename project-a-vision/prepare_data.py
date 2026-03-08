import json
import os
from PIL import Image
from datasets import Dataset, Image as HFImage

# 1. 설정
input_file = "./data/dataset.jsonl"
output_dir = "./mlx_ready_dataset"
os.makedirs(output_dir, exist_ok=True)

data = []

print("🚀 완벽한 Parquet 데이터셋으로 변환을 시작합니다...")

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        
        # 경로 추출
        img_rel_path = item["images"][0] if isinstance(item["images"], list) else item["images"]
        full_path = os.path.abspath(os.path.join("./data", img_rel_path))
        
        try:
            # 핵심 1: 이미지를 직접 열어버림
            img = Image.open(full_path).convert("RGB")
            
            # 핵심 2: 이름은 'images'지만, 리스트가 아닌 '단일 객체(img)'로 삽입!
            data.append({
                "images": img,  
                "messages": item["messages"]
            })
        except Exception as e:
            print(f"❌ 건너뜀 (이미지 에러): {full_path} -> {e}")

# 3. HF 데이터셋으로 변환
ds = Dataset.from_list(data)

# 4. 'images' 컬럼을 진짜 Image 타입으로 강제 캐스팅
ds = ds.cast_column("images", HFImage())

# 5. Parquet 포맷으로 저장 (mlx_vlm이 가장 완벽하게 읽는 포맷)
output_file = os.path.join(output_dir, "train.parquet")
ds.to_parquet(output_file)

print(f"✅ 변환 완료! 최종 데이터셋: {output_file}")