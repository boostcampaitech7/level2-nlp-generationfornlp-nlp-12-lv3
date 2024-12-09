import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from pandas.core.api import DataFrame as DataFrame
from pipeline import GenPipeline
from manager import ModelManager
from transformers import StoppingCriteriaList
from .utils import StopOnAnswer
from typing import Type, Optional

class CoTPipeline(GenPipeline):
    def __init__(self, config_name: str, Manager: Type[ModelManager],
                 answer_template: Optional[str] = None): 
        super().__init__(config_name, Manager)
        if answer_template is None:
            self.answer_template = "Therefore, the following number of choice is the correct answer: {answer}"
    
    def make_chat_message(self, row: dict, user_message: str) -> dict:
        return {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "label": f"{row['reason']}\n\n" + self.answer_template.format(answer=row["answer"]), 
        }
    
    def simple_parse(self, pred: str) -> int:
        x = 0
        for i in range(len(pred), 0, -1):
            if pred[i-1] in ["1", "2", "3", "4", "5"]:
                x = int(pred[i-1])
                break
        return x
    
    def compute_metrics(self, eval_preds):
        metrics = super().compute_metrics(eval_preds) # get BLEU
        
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
        
        parsed_preds = [self.simple_parse(pred) for pred in decoded_preds]
        parsed_labels = [self.simple_parse(label) for label in decoded_labels]
        accuracy = sum([1 for pred, label in zip(parsed_preds, parsed_labels) if pred == label]) / len(parsed_preds)
        metrics["eval_accuracy"] = accuracy
        
        return metrics

    def do_inference(self, model: AutoPeftModelForCausalLM, dataset: DataFrame) -> DataFrame:
        tokenizer = self.manager.tokenizer
        stop_criteria = StopOnAnswer(tokenizer, self.answer_template)
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
                
                infer_results.append({"id": _id, "reason": outputs, "reason": self.simple_parse(outputs)})
                self.save_df(pd.DataFrame(infer_results), "tmp.csv")
        return pd.DataFrame(infer_results)