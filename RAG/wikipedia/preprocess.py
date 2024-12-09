import os
import json
from tqdm import tqdm
import pandas as pd
import re
import html

# 데이터 디렉토리 설정
extracted_dir = "extracted"  # wikiextractor 출력 디렉토리

# JSON 데이터 로드 함수
def load_extracted_data_json_lines(extracted_dir):
    data = []
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.startswith("wiki"):  # wiki_* 파일만 처리
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:  # JSON 라인별로 읽기
                        try:
                            document = json.loads(line)
                            if "text" in document and document["text"].strip():  # 유효한 텍스트만 추가
                                data.append({
                                    "id": document["id"],
                                    "url": document["url"],
                                    "title": document["title"],
                                    "content": document["text"]
                                })
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {e}")
    return data

# 텍스트 클리닝
def clean_text(text):
    # 1. HTML 언어 기호 복원 (예: &amp; -> &)
    text = html.unescape(text)

    # 2. 특수 기호 및 formula_* 같은 패턴 제거
    text = re.sub(r"formula_\d+", "", text)  # formula_숫자 제거
    text = re.sub(r"[^\w\s.,!?가-힣]", " ", text)  # 알파벳, 숫자, 한글, 공백, 기본 특수문자 제외

    # 3. 불필요한 공백 제거
    text = re.sub(r"\s+", " ", text).strip()  # 여러 공백을 하나로 압축, 양쪽 공백 제거
    return text

# 데이터 로드 및 처리
print("Loading and processing JSON lines data...")
wiki_data = load_extracted_data_json_lines(extracted_dir)

# 텍스트 클리닝
for doc in tqdm(wiki_data, desc="Cleaning Text"):
    doc["content"] = clean_text(doc["content"])

# JSON 저장
output_json = "cleaned_wikipedia.json"
print(f"Saving cleaned data to {output_json}...")
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(wiki_data, f, ensure_ascii=False, indent=4)

# CSV 저장
output_csv = "cleaned_wikipedia.csv"
print(f"Saving cleaned data to {output_csv}...")
df = pd.DataFrame(wiki_data)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("Processing completed successfully!")
