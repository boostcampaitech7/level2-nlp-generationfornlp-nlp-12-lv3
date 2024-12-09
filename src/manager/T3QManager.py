# import os
import torch
from transformers import AutoTokenizer 
from trl import DataCollatorForCompletionOnlyLM
from manager import MistralManager
from functools import partial

class T3QManager(MistralManager):
    def set_data_collator(self):
        response_template = "<|im_start|>assistant\n"

        def find_subsequence(sequence, subsequence):
            """시퀀스에서 하위 시퀀스의 시작 인덱스를 찾는 헬퍼 함수."""
            for idx in range(len(sequence) - len(subsequence) + 1):
                if sequence[idx:idx + len(subsequence)] == subsequence:
                    return idx
            return -1  # 찾을 수 없을 때 -1 반환
        
        def custom_data_collator(features, tokenizer, response_template):
            for feature in features:
                input_ids = feature['input_ids']
                labels = [-100] * len(input_ids)  # 모든 라벨을 기본적으로 무시(-100)로 초기화
                
                # response_template을 토큰화
                response_template_tokens = tokenizer.encode(response_template, add_special_tokens=False) 
                
                 # print("response_tokens:", response_template_tokens)
                # print("decoded_tokens: ", [tokenizer.decode(token) for token in response_template_tokens])
                # print("last 15 tokens: ", input_ids[-15:])
                # print("decoded_tokens: ", [tokenizer.decode(token) for token in input_ids[-15:]])
                response_template_tokens[0] = 28789
                
                # input_ids 내에서 response_template의 시작 인덱스 찾기
                template_start_idx = find_subsequence(input_ids, response_template_tokens)
                start_token_idx = template_start_idx + len(response_template_tokens)
                
                # response_template 이후 텍스트 추출
                extracted_output_tokens = input_ids[start_token_idx:]  # 응답 부분만 추출
                extracted_output = tokenizer.decode(extracted_output_tokens, skip_special_tokens=True)
                
                # 추출된 출력 내용 출력
                print("Extracted Output:", extracted_output)
                
                if template_start_idx != -1:
                    # response_template 이후 응답의 시작 토큰 인덱스 계산
                    start_token_idx = template_start_idx + len(response_template_tokens)
                    
                    # response_template 이후의 모든 토큰을 라벨로 설정
                    for i in range(start_token_idx, len(input_ids)):
                        labels[i] = input_ids[i]
                else:
                    print("Response template이 input_ids 내에서 발견되지 않았습니다.")
                
                feature['labels'] = labels  # 라벨 설정
                
            return tokenizer.pad(features, return_tensors="pt")  # 배치 패딩 처리
        self.data_collator = partial(custom_data_collator, tokenizer=self.tokenizer, response_template=response_template)
        
    def set_tokenizer(self):
        """토크나이저를 불러옵니다. 토크나이저는 self.tokenizer에 할당합니다.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True,
            use_fast = True
        )
        if self.TEMPLATE is not None:
            self.tokenizer.chat_template = self.TEMPLATE
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'