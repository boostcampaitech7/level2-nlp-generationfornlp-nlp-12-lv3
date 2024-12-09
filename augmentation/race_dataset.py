import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class M2MTranslator:
    def __init__(self, model_name="facebook/m2m100_1.2B", src_lang="en", tgt_lang="ko"):
        """
        M2M 번역 모델 초기화
        
        Args:
            model_name (str): 사용할 M2M 모델 이름
            src_lang (str): 소스 언어 코드
            tgt_lang (str): 타겟 언어 코드
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # 소스 및 타겟 언어 설정
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

    def translate_batch(self, texts, max_length=512):
        """
        텍스트 배치 번역
        
        Args:
            texts (list): 번역할 텍스트 리스트
            max_length (int): 최대 시퀀스 길이
        
        Returns:
            list: 번역된 텍스트 리스트
        """
        # 빈 리스트 처리
        if not texts:
            return []
        
        # 텍스트 전처리 (None 값 제거, 문자열로 변환)
        texts = [str(text) for text in texts if text is not None]
        
        if not texts:
            return []
        
        # 토크나이징
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        # 번역 생성
        translated = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang)
        )
        
        # 디코딩
        translations = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translations

def process_race_dataset(batch_size=8):
    """
    RACE 데이터셋 번역 및 처리
    
    Args:
        batch_size (int): 배치 크기
    
    Returns:
        pandas.DataFrame: 번역된 데이터프레임
    """
    # 데이터셋 로드
    dataset = load_dataset("ehovy/race", "all")
    dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

    def custom_collate_fn(batch):
        # 각 필드를 개별적으로 처리
        articles = [item['article'] for item in batch]
        questions = [item['question'] for item in batch]
        options = [item['options'] for item in batch]
        answers = [item['answer'] for item in batch]
        example_ids = [item['example_id'] for item in batch]
        
        return {
            "article": articles,
            "question": questions,
            "options": options,
            "answer": answers,
            "example_id": example_ids
        }

    # 데이터셋 슬라이싱 방법 (처음 3000개 선택)
    subset_dataset = Subset(dataset, indices=range(3000))

    # DataLoader 생성
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    translator = M2MTranslator()
    dataframe = []
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_lengths = [len(batch[field]) for field in ['article', 'question', 'options', 'answer', 'example_id']]
        if len(set(batch_lengths)) != 1:
            raise ValueError(f"Inconsistent batch lengths: {batch_lengths}")

        batch_lengths = batch_lengths[0]

        paragraphs = batch['article']
        questions = batch['question']
        options_batches = batch['options']
        answers = batch['answer']
        example_ids = batch['example_id']
        
        # 배치 번역
        translated_paragraphs = translator.translate_batch(paragraphs)
        translated_questions = translator.translate_batch(questions)
        
        # 옵션 배치 번역 (중첩 리스트 처리)
        translated_options_batches = []
        for options in options_batches:
            translated_options = translator.translate_batch(options)
            translated_options_batches.append(translated_options)
        
        # 결과 처리
        for j in range(batch_lengths):
            row = {"question_id": example_ids[j]}
            
            row["paragraph"] = translated_paragraphs[j]
            
            # 답변 인코딩
            answer = answers[j]
            if answer == "A":
                answer = 1
            elif answer == "B":
                answer = 2
            elif answer == "C":
                answer = 3
            elif answer == "D":
                answer = 4
            
            row["problems"] = {
                "question": translated_questions[j],
                "choices": translated_options_batches[j],
                "answer": answer
            }
            
            row["question_plus"] = ""
            
            dataframe.append(row)

    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(dataframe)
    df.to_csv("data/race_data_m2m_2.csv", index=False, encoding='utf-8-sig')
    
    return df

# 메인 실행
if __name__ == "__main__":
    translated_df = process_race_dataset()
    print(f"Translated {len(translated_df)} samples")