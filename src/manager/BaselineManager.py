from typing import Type, Union
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from manager import ModelManager

class BaselineManager(ModelManager):
    def __init__(self, model_config, params):
        super().__init__(model_config, params)
        self.TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"

    def set_model(self, AutoModel: Union[Type[AutoModelForCausalLM], Type[AutoPeftModelForCausalLM]] = AutoModelForCausalLM):
        self.model = AutoModel.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True,
            device_map="auto"
        )
    