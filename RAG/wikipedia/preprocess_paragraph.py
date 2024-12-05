import json
import re
from tqdm import tqdm
import pandas as pd
import html
import os

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

    # 2. 소제목 제거 (예: == 소제목 ==)
    text = re.sub(r"^==.*?==\s*$", "", text, flags=re.MULTILINE)  # 소제목 패턴 제거

    # 3. 특수 기호 및 formula_* 같은 패턴 제거
    text = re.sub(r"formula_\d+", "", text)  # formula_숫자 제거
    text = re.sub(r"[^\w\s.,!?가-힣]", " ", text)  # 알파벳, 숫자, 한글, 공백, 기본 특수문자 제외

    # 4. 불필요한 공백 제거
    text = re.sub(r"[ \t]+", " ", text)  # 공백만 압축
    return text.strip()

# 문단 단위로 분리
def split_into_paragraphs(text):
    paragraphs = text.split("\n")  # 줄바꿈을 기준으로 분리
    paragraphs = [p.strip() for p in paragraphs if p.strip()]  # 빈 문단 제거
    return paragraphs

def is_valid_paragraph(text):
    # 최소 길이 체크 (예: 50자 이상)
    if len(text) < 50:
        return False
    
    # 마침표로만 끝나는 한 단어 문단 체크
    if re.match(r'^[가-힣A-Za-z]+\.$', text.strip()):
        return False
    
    # 소제목 형태 체크
    if text.endswith('.') and len(text.split()) <= 2:
        return False
        
    return True

# 데이터 로드 및 처리
print("Loading and processing JSON lines data...")
wiki_data = load_extracted_data_json_lines(extracted_dir)

# 텍스트 클리닝 및 문단 분리
processed_data = []
for doc in tqdm(wiki_data, desc="Processing documents"):
    cleaned_content = clean_text(doc["content"])
    paragraphs = split_into_paragraphs(cleaned_content)
    for i, paragraph in enumerate(paragraphs):
        # 유효한 문단만 추가
        if is_valid_paragraph(paragraph):
            processed_data.append({
                "id": f"{doc['id']}_{i}",
                "url": doc["url"],
                "title": doc["title"],
                "paragraph": paragraph
            })

# JSON 저장
output_json = "cleaned_wikipedia_paragraphs.json"
print(f"Saving cleaned data to {output_json}...")
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

# CSV 저장
output_csv = "cleaned_wikipedia_paragraphs.csv"
print(f"Saving cleaned data to {output_csv}...")
df = pd.DataFrame(processed_data)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print("Processing completed successfully!")
