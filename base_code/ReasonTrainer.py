from ast import literal_eval

from trainer import MyTrainer
from config.prompts import PROMPT_GEN_REASON_NO_QUESTION_PLUS, PROMPT_GEN_REASON_QUESTION_PLUS

import pandas as pd
import numpy as np
from datasets import Dataset
from evaluate import load

class ReasonTrainer(MyTrainer):
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """입력으로 들어온 dataset을 이용하여 입력 프롬프트와 그에 대한 정답을 생성합니다.

        Args:
            dataset (Dataset): 입력으로 들어온 dataset. 아래 필드를 포함하고 있어야 합니다.
            (id-식별자, 지문-paragraph, 질문-question, 선지-choices, 보기-question_plus, 정답-answer
             reason-해설)
        
        Returns:
            Dataset: 아래 필드를 포함하고 있는 dataset을 반환합니다.
            (id-식별자, messages-프롬프트를 위한 message, label-정답)
        """ 
        processed_dataset = []
        for i in range(len(dataset)):
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])
            # <보기>가 있을 때
            if dataset[i]["question_plus"]:
                user_message = PROMPT_GEN_REASON_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = PROMPT_GEN_REASON_NO_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    choices=choices_string,
                )
            # chat message 형식으로 변환
            processed_dataset.append(
                {
                    "id": dataset[i]["id"],
                    "messages": [
                        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": f"{dataset[i]['reason']}"}
                    ],
                    "label": dataset[i]['reason'],
                }
            )
            
        return Dataset.from_pandas(pd.DataFrame(processed_dataset))

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
        
        def find_non_pad(row):
            return np.argmax(row != -100)
        answer_indices = np.apply_along_axis(find_non_pad, axis=-1, arr=labels)
        
        for i, idx in enumerate(answer_indices):
            predictions[i, :idx] = -100
        labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
        predictions = np.where(predictions == -100, self.tokenizer.pad_token_id, predictions)
  
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
  
        result = metric.compute(
            predictions=[pred for pred in decoded_preds],  # 토큰화된 예측값
            references=[label for label in decoded_labels]  # 토큰화된 참조값
        )
  
        return result
  
    def load_dataset(self) -> Dataset:
        df = pd.read_csv(self.data_path + "/train.csv")
        df = df[df['reason'].notnull()]
        records = []
        for _, row in df.iterrows():
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
        return Dataset.from_pandas(pd.DataFrame(records))
    
    def report_metrics(self, metrics):
        print("-" * 30)
        print("BLEU Score: ", metrics["eval_bleu"])
        print("-" * 30)


if __name__ == "__main__":
    import argparse
    from utils.load import load_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config", help="path where config json is store")
    args = parser.parse_args()
    
    config = load_config(args.config)
    settings = config["settings"]
    trainer = ReasonTrainer(settings["dataset"], settings["model_name"], config["params"])
    trainer.train()