import torch
from typing import Type, Union
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from manager import ModelManager

class EXAONEManager(ModelManager):
    def set_model(self, AutoModel: Union[Type[AutoModelForCausalLM], Type[AutoPeftModelForCausalLM]] = AutoModelForCausalLM):
        self.model = AutoModel.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
    def set_data_collator(self):
        response_template = "[|assistant|]"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )