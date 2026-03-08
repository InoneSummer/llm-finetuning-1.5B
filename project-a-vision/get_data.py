import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# 1. 설정
DATASET_NAME = "ronantakizawa/webui"
SAVE_DIR = "./refined_webui_data"
IMAGE_SUBDIR = "images"
JSONL_FILENAME = "dataset.jsonl"
USER_PROMPT = "Convert this web UI screenshot into functional HTML and CSS code."

# 폴더 생성
os.makedirs(os.path.join(SAVE_DIR, IMAGE_SUBDIR), exist_ok=True)

# 2. 데이터셋 로드 (학습용 데이터만 500개 샘플링 - 처음엔 작게 시작 권장)
print(f"🚀 {DATASET_NAME} 데이터셋 로딩 중...")
dataset = load_dataset(DATASET_NAME, split="train", streaming=False)

# 3. 변환 및 저장
print("📸 이미지 저장 및 JSONL 생성 시작...")
with open(os.path.join(SAVE_DIR, JSONL_FILENAME), "w", encoding="utf-8") as f:
    # 너무 많으면 학습이 오래 걸리므로 우선 300개만 정제해서 사용해 보세요.
    for i, item in enumerate(tqdm(dataset.select(range(min(len(dataset), 300))))):
        try:
            # 파일명 설정
            img_filename = f"sample_{i:04d}.jpg"
            img_path = os.path.join(IMAGE_SUBDIR, img_filename)
            full_img_path = os.path.join(SAVE_DIR, img_path)

            # 이미지 저장 (PIL Image 객체인 경우)
            image = item['image']
            if not isinstance(image, Image.Image):
                # 만약 경로로 되어 있다면 열어서 저장
                image = Image.open(image)
            
            image.convert("RGB").save(full_img_path, "JPEG")

            # HTML 코드 가져오기 (컬럼명이 'code' 또는 'text'인지 확인 필요)
            # 해당 데이터셋의 구조에 따라 'code' 또는 'html'로 변경하세요.
            html_code = item.get('code', item.get('html', ""))
            
            if not html_code:
                continue

            # MLX-VLM 포맷으로 구성
            entry = {
                "images": [img_path],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {"type": "image"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": html_code}
                        ]
                    }
                ]
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        except Exception as e:
            print(f"❌ {i}번째 데이터 처리 중 에러: {e}")

print(f"✅ 모든 작업 완료! 데이터 위치: {SAVE_DIR}")