import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from pandas.core.api import DataFrame as DataFrame
from peft import AutoPeftModelForCausalLM
from pipeline import GenPipeline
from vllm import LLM, SamplingParams

class VLLMPipeline(GenPipeline):
    def call_vllm(self, model: AutoPeftModelForCausalLM, model_name_or_checkpoint: str):
        checkpoint = model_name_or_checkpoint.split("/")[-1]
        save_path = f"{self.manager.params.output_dir}/merged-{checkpoint}"
        if not os.path.exists(save_path):
            merged = model.merge_and_unload()
            merged.save_pretrained(save_path)
        
        self.manager.model.to("cpu")
        del self.manager.model
        torch.cuda.empty_cache()
    
        self.manager.model = LLM(
            model=save_path, 
            tokenizer=model_name_or_checkpoint,
            trust_remote_code=True,
        )
        
        return self.manager.model 
    
    def do_inference(self, model: AutoPeftModelForCausalLM, dataset: Dataset) -> DataFrame:
        max_tokens = self.manager.model.config.max_position_embeddings
        llm = self.call_vllm(model, self.manager.model_name_or_checkpoint)
        infer_results = []
        tokenizer = self.manager.tokenizer
        
        params = SamplingParams(
            max_tokens = max_tokens,
        )
        
        with torch.inference_mode():
            for data in tqdm(dataset):
                _id = data["id"]
                messages = data["messages"]
                
                results = llm.generate(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False
                    ),
                    params
                )
                outputs = results[0].outputs[0].text
                print(outputs)
                infer_results.append({"id": _id, "reason": outputs})
        return pd.DataFrame(infer_results)