from abc import ABC, abstractmethod
from transformers import AutoTokenizer, Trainer, AutoModelForCausalLM
from datasets import Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from typing import Union, Type, List


class ModelManager(ABC):
    def __init__(self, model_config, params):
        self.model_config = model_config
        self.model_name_or_checkpoint = model_config["name"]
        self.params = params

        self.TEMPLATE = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_collator = None

    def formatting_prompts_func(self, example: dict) -> List[dict]:
        """입력으로 들어온 example을 이용하여 입력 프롬프트를 생성합니다.

        Args:
            example (dict): "messages" 필드를 포함해야 합니다.

        Returns:
            dict: 입력 프롬프트를 반환합니다.
        """
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False
                )
            )
        return output_texts

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
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {
            "id": element["id"],
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    @abstractmethod
    def set_model(self, AutoModel: Union[Type[AutoModelForCausalLM], Type[AutoPeftModelForCausalLM]] = AutoModelForCausalLM):
        """모델을 불러옵니다. 모델은 self.model에 할당합니다.

        Args:
            AutoModel (Union[Type[AutoModelForCausalLM], Type[AutoPeftModelForCausalLM]]): 
            - Type[AutoModelForCasualLM]: 훈련 단계에서 사용합니다.
            - Type[AutoPeftModelForCausalLM]: LoRA가 학습된 모델을 불러올 때 사용합니다. (즉, 추론 단계)
        """
        pass

    def get_model(self) -> Union[AutoModelForCausalLM, AutoPeftModelForCausalLM]:
        """모델을 반환합니다. (Pipeline 추론 단계에서 사용)

        Returns:
            Union[AutoModelForCausalLM, AutoPeftModelForCausalLM]: 모델을 반환합니다.
        """
        return self.model

    def set_tokenizer(self):
        """토크나이저를 불러옵니다. 토크나이저는 self.tokenizer에 할당합니다.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True
        )
        if self.TEMPLATE is not None:
            self.tokenizer.chat_template = self.TEMPLATE
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'

    def set_data_collator(self):
        """DataCollator 객체를 생성합니다. 객체는 self.data_collator에 할당합니다.
        """
        response_template = "<start_of_turn>model"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

    def set_trainer(self, train_dataset: Dataset, eval_dataset: Dataset, compute_metrics, preprocess_logits_for_metrics) -> Trainer:
        """Trainer 객체를 생성합니다. 다음 인스턴스 변수들이 존재해야 합니다.
        - model: 훈련시킬 모델
        - train_dataset: 훈련 데이터셋
        - eval_dataset: 검증 데이터셋
        - data_collator: 데이터 콜레이터 함수
        - tokenizer: 토크나이저
        - params: 훈련에 필요한 하이퍼파라미터

        Args:
            train_dataset (Dataset): 훈련 데이터셋
            eval_dataset (Dataset): 검증 데이터셋
            compute_metrics (Callable): 평가 지표를 계산하는 함수
            preprocess_logits_for_metrics (Callable): 평가 지표 계산을 위한 logits 전처리 함수

        Returns:
            Trainer: Trainer 객체를 반환합니다.
        """
        peft_config = LoraConfig(
            r=self.params["r"],
            lora_alpha=self.params["lora_alpha"],
            lora_dropout=self.params["lora_dropout"],
            target_modules=['q_proj', 'k_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        sft_config = SFTConfig(
            do_train=True,
            do_eval=self.params["do_eval"],
            lr_scheduler_type=self.params["lr_scheduler_type"],
            max_seq_length=self.params["max_seq_length"],
            output_dir=self.params["output_dir"],
            per_device_train_batch_size=self.params["train_batch_size"],
            gradient_accumulation_steps=self.params["gradient_accumulation_steps"],
            per_device_eval_batch_size=self.params["eval_batch_size"],
            num_train_epochs=self.params["epoch"],
            learning_rate=float(self.params["learning_rate"]),
            warmup_ratio=self.params["warmup_ratio"],
            weight_decay=self.params["weight_decay"],
            logging_steps=self.params["logging_steps"],
            save_strategy=self.params["save_strategy"],
            eval_strategy=self.params["eval_strategy"],
            eval_steps=self.params["eval_steps"],
            fp16=self.params["fp16"],
            fp16_full_eval=self.params["fp16_full_eval"],
            load_best_model_at_end=self.params["load_best_model_at_end"],
            metric_for_best_model=self.params["metric_for_best_model"],
            greater_is_better=self.params["greater_is_better"],
            save_total_limit=2,
            save_only_model=True,
            report_to="none",
            log_level='error'
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            # compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            args=sft_config
        )

        print("-" * 30)
        print("lr_scheduler_type : {}".format(
            self.params["lr_scheduler_type"]))
        print("max_seq_length : {}".format(self.params["max_seq_length"]))
        print("train_batch_size : {}".format(self.params["train_batch_size"]))
        print("eval_batch_size : {}".format(self.params["eval_batch_size"]))
        print("epoch : {}".format(self.params["epoch"]))
        print("learning_rate : {}".format(self.params["learning_rate"]))
        print("weight_decay : {}".format(self.params["weight_decay"]))
        print("-" * 30)

    def train(self, *args, **kwargs):
        """모델을 훈련합니다.
        """
        self.trainer.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> dict:
        """모델을 평가합니다.

        Returns:
            dict: 평가 지표를 반환합니다.
        """
        return self.trainer.evaluate(*args, **kwargs)
