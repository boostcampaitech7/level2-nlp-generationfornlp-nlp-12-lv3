import numpy as np
from ast import literal_eval
from evaluate import load
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from peft import AutoPeftModelForCausalLM 
from .utils import StopOnText
from transformers import StoppingCriteriaList

from pipeline import BasePipeline 
from .prompts import PROMPT_GEN_REASON_QUESTION_PLUS, PROMPT_GEN_REASON_NO_QUESTION_PLUS

class GenPipeline(BasePipeline):
    
    def make_user_messages(self, row: dict) -> str:
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            # <보기>가 있을 때
        if row["question_plus"]:
            user_message = PROMPT_GEN_REASON_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_GEN_REASON_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )
            
        return user_message
    
    def make_chat_message(self, row: dict, user_message: str) -> dict:
        return {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "label": row['reason'],
        }


    # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    def preprocess_logits_for_metrics(self, logits, labels):
        # 로짓에서 가장 높은 확률의 토큰 인덱스를 추출 (argmax)
        predictions = logits.argmax(dim=-1)
        # print(predictions)
        return predictions.cpu(), labels.cpu()

    def compute_metrics(self, eval_preds):
        metric = load("bleu")  # BLEU 평가 지표
        predictions, labels = eval_preds
        predictions = predictions[0] # predictions = Tuple[np.ndarray]
        tokenizer = self.manager.tokenizer
        
        def find_non_pad(row):
            return np.argmax(row != -100)
        answer_indices = np.apply_along_axis(find_non_pad, axis=-1, arr=labels)
        
        for i, idx in enumerate(answer_indices):
            predictions[i, :idx] = -100
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
  
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  
        result = metric.compute(
            predictions=[pred for pred in decoded_preds],  # 토큰화된 예측값
            references=[label for label in decoded_labels]  # 토큰화된 참조값
        )
  
        return result
  
    @classmethod
    def load_dataset(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        records = []
        if "reason" not in dataset.columns:
            dataset["reason"] = ""
        dataset = dataset[dataset["reason"].notnull()]
        for _, row in dataset.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
                "reason": row["reason"]
            }
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
        return pd.DataFrame(records)
    
    def report_metrics(self, metrics):
        print("-" * 30)
        print("BLEU Score: ", metrics["eval_bleu"])
        print("-" * 30)

    def do_inference(self, model: AutoPeftModelForCausalLM, dataset: Dataset) -> pd.DataFrame:
        tokenizer = self.manager.tokenizer
        stop_criteria = StopOnText(tokenizer, '<end_of_turn>')
        stopping_criteria = StoppingCriteriaList([stop_criteria])
        
        infer_results = []
        model.eval()
        with torch.inference_mode():
            for data in tqdm(dataset):
                _id = data["id"]
                messages = data["messages"]

                chat = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.device)
                
                outputs = model.generate(
                    input_ids = chat,
                    max_length = model.config.max_position_embeddings,
                    stopping_criteria=stopping_criteria
                ) 
                
                outputs = tokenizer.decode(outputs
                    .detach()
                    .cpu()
                    .numpy()[0])
                print(outputs)
                
                infer_results.append({"id": _id, "reason": outputs})
        return pd.DataFrame(infer_results)
    
    def inference(self) -> pd.DataFrame:
        output = super().inference()
        if self.manager.data_collator is None:
            self.manager.set_data_collator()
        test_df = pd.read_csv(self.data_path + "/" + self.data_config["test_file"])
        output = output[output['reason'].notnull()]
        def get_only_cot(x):
            x = x.split(self.manager.data_collator.response_template)[-1]
            x = x.replace(self.manager.tokenizer.eos_token, "")
            return x
        
        test_df['reason'] = output['reason'].apply(get_only_cot)
        self.save_df(test_df, "output-postprocess.csv")
        return test_df