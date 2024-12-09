# import os
import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from functools import partial
from manager import ModelManager
from typing import Type, Union
from peft import AutoPeftModelForCausalLM
from bitsandbytes import BitsAndBytesConfig
from transformers import AutoModelForCausalLM


class EEVEManager(ModelManager):
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
        response_template = "<|im_start|>assistant\n"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

    def set_tokenizer(self):
        """토크나이저를 불러옵니다. 토크나이저는 self.tokenizer에 할당합니다.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True,
            use_fast=True
        )
        if self.TEMPLATE is not None:
            self.tokenizer.chat_template = self.TEMPLATE
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
