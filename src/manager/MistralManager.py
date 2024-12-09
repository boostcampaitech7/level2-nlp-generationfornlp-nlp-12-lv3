# import os
import torch
from trl import DataCollatorForCompletionOnlyLM
from typing import Type, Union
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from manager import ModelManager

class MistralManager(ModelManager):
    def set_model(self, AutoModel: Union[Type[AutoModelForCausalLM] | Type[AutoPeftModelForCausalLM]] = AutoModelForCausalLM):
        
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # QLoRA에서 권장하는 nf4 사용
            bnb_4bit_compute_dtype=torch.bfloat16,  # 메모리 효율을 위해 bfloat16 사용
            bnb_4bit_use_double_quant=True,  # QLoRA에서 double quantization 사용
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_name_or_checkpoint,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="offload"
        )
        
        for name, param in self.model.named_parameters():
            if "transformer.h.0" in name:  # 예: 초기 레이어를 동결
                param.requires_grad = False
    
    def set_data_collator(self):
        response_template = "[/INST]"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )
        
    def apply_chat_template_and_tokenize(self, element: dict) -> dict:
        """데이터셋의 각 element를 프롬프트로 변환하고 토크나이징합니다.

        Args:
            element (dict): "messages" 필드를 포함해야 합니다.

        Returns:
            dict: 토크나이징된 결과를 반환합니다.
            - input_ids: 토큰화된 input ids
            - attention_mask: 토큰화된 attention mask
        """
        outputs = self.tokenizer(
            self.formatting_prompts_func(element),
            truncation=True,
            padding=False,
            max_length=2048,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {
            "id": element["id"],
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }