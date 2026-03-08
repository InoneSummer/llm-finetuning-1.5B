import json
import os
from pathlib import Path

# 설정
base_dir = Path("data/processed")
files_to_fix = ["train.jsonl", "val.jsonl"]
system_prompt = "You are an expert UI developer. Given a screenshot of a UI, generate clean and functional HTML/CSS code that accurately reproduces the layout, components, and visual style shown."

def convert_dataset():
    for filename in files_to_fix:
        file_path = base_dir / filename
        if not file_path.exists():
            print(f"⚠️ {filename} 파일을 찾을 수 없어 건너뜁니다.")
            continue
            
        print(f"🔄 {filename} 변환 및 메모리 최적화 중...")
        fixed_data = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 메시지 추출 (기존 키가 무엇이든 대응)
                    raw_msgs = data.get("messages") or data.get("conversations") or []
                    
                    new_messages = []
                    # 시스템 프롬프트를 유저 메시지에 통합하여 토큰 계산 안정화
                    user_content = ""
                    assistant_content = ""
                    
                    for msg in raw_msgs:
                        role = msg.get("role") or ("user" if msg.get("from") == "human" else "assistant")
                        content = msg.get("content") or msg.get("value") or ""
                        
                        if role == "user":
                            # 태그 정제 및 줄바꿈 제거 (Shape 에러 방지 핵심)
                            clean_content = content.replace("<image>", "<|image|>").strip()
                            user_content = f"<|image|>{clean_content}
                        elif role == "assistant":
                            assistant_content = content.strip()

                    new_messages.append({"role": "user", "content": user_content})
                    new_messages.append({"role": "assistant", "content": assistant_content})

                    fixed_data.append({
                        "images": data.get("images") or data.get("image"), # MLX-VLM 요구사항
                        "messages": new_messages
                    })
                except Exception as e:
                    print(f"❌ 에러 발생: {e}")

        # 변환된 내용 저장
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in fixed_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 모든 데이터셋 변환 완료! (경로: {base_dir})")

if __name__ == "__main__":
    convert_dataset()