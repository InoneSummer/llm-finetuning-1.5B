from datasets import load_dataset
import os

print("✂️ 데이터셋 다이어트 (텍스트 길이 자르기) 시작...")

# 1. 기존 파케이 데이터셋 불러오기
ds = load_dataset("parquet", data_files="./mlx_ready_dataset/train.parquet")["train"]

def truncate(example):
    new_msgs = []
    for msg in example["messages"]:
        text = msg["content"]
        
        # 2. 코드(텍스트) 길이가 3000자를 넘어가면 강제 절단!
        # 여기서 메모리 폭발이 일어났던 것입니다.
        if isinstance(text, str) and len(text) > 3000:
            text = text[:3000] + "\n"
            
        new_msgs.append({"role": msg["role"], "content": text})
    
    example["messages"] = new_msgs
    return example

# 3. 데이터셋 전체에 매핑 적용
truncated_ds = ds.map(truncate)

# 4. 새로운 폴더에 저장
output_dir = "./mlx_ready_dataset_truncated"
os.makedirs(output_dir, exist_ok=True)
truncated_ds.to_parquet(f"{output_dir}/train.parquet")

print(f"✅ 완료! 다이어트된 데이터셋이 다음 폴더에 생성되었습니다: {output_dir}")