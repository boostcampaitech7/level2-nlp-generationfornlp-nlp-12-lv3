import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from pipeline import BasePipeline

class BaseLogitPipeline(BasePipeline):
    
    def do_inference(self, model: AutoPeftModelForCausalLM, dataset: Dataset) -> pd.DataFrame:
        infer_results = []
        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        tokenizer = self.manager.tokenizer
        model.eval()
        with torch.inference_mode():
            for data in tqdm(dataset):
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                outputs = model(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(self.device)
                )

                logits = outputs.logits[:, -1].flatten().cpu()

                target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
                
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(target_logit_list, dtype=torch.float32)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
                
                row = {"id": _id, "answer": predict_value}
                
                
                target_logit_list = [logit.item() for logit in target_logit_list]
                if len_choices < len(pred_choices_map):
                    target_logit_list += [None] * (len(pred_choices_map) - len_choices)
                for i, logit in enumerate(target_logit_list):
                    row[f"logit_{pred_choices_map[i]}"] = logit
                
                infer_results.append(row)
                
        return pd.DataFrame(infer_results)
    
    def inference(self) -> pd.DataFrame:
        output = super().inference()
        postprocessed_output = output[["id", "answer"]]
        self.save_df(postprocessed_output, "output-postprocess.csv")
        return postprocessed_output
    