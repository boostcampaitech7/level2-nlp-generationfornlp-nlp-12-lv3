import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import evaluate
from peft import AutoPeftModelForCausalLM
from datasets import Dataset
from ast import literal_eval
from .utils import load_config, load_last_commit
from .prompts import PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS
from manager import ModelManager
from typing import Type, Tuple, Optional

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BasePipeline:
    def __init__(self, config_name: str, Manager: Type[ModelManager]):
        """BasePipeline 클래스의 생성자입니다.

        Args:
            config_name (str): config 파일의 경로 (확장자 제외)
        """
        config = load_config(config_name)

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config_name = config_name.split("/")[-1]

        self.data_config = config["data"]
        self.data_path = self.data_config["data_path"]

        self.experiment_config = config["experiment"]
        self.output_path = self.experiment_config.get("output_dir", ".")

        model_config = config["model"]
        params = config["params"]
        self.manager = Manager(model_config, params)

        config["pipeline"] = self.__class__.__name__
        config["manager"] = self.manager.__class__.__name__
        config["version"] = load_last_commit()
        self.save_json(config, "config.json")

    def save_json(self, x: dict, file_path: str):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(f"{self.output_path}/{self.config_name}"):
            os.makedirs(f"{self.output_path}/{self.config_name}")
        with open(f"{self.output_path}/{self.config_name}/{file_path}", "w") as f:
            json.dump(x, f, indent=4)

    def save_df(self, df: pd.DataFrame, file_path: str):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(f"{self.output_path}/{self.config_name}"):
            os.makedirs(f"{self.output_path}/{self.config_name}")
        df.to_csv(
            f"{self.output_path}/{self.config_name}/{file_path}", index=False)

    def preprocess_logits_for_metrics(self, logits, labels):
        tokenizer = self.manager.tokenizer
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"],
                     tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
        return logits

    @classmethod
    def load_dataset(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임을 로드하여 필요한 필드를 추가하여 데이터프레임을 반환합니다.

        Args:
            dataset (pd.DataFrame): "id", "paragraph", "problems" 필드를 포함해야 합니다.

        Returns:
            pd.DataFrame: "id", "paragraph", "question", "choices", "answer" 필드를 포함한 데이터프레임을 반환합니다.
        """
        records = []
        for _, row in dataset.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
            }
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
        df = pd.DataFrame(records)
        return df

    def _load_dataset(self, mode: str = "train", df: Optional[pd.DataFrame] = None) -> Dataset:
        """
        csv 파일을 로드하여 허깅페이스 Dataset 형태로 반환합니다.

        Args:
            mode (str): 데이터셋 모드 (train, dev, test 중 하나)
            df (pd.DataFrame): 데이터프레임을 직접 입력할 경우 사용합니다.

        Returns:
            Dataset: 데이터셋
        """

        file_name = None
        if df is not None:
            return Dataset.from_pandas(self.load_dataset(df))
        elif mode == "test":
            file_name = self.data_config["test_file"]
        elif mode == "dev":
            file_name = self.data_config["dev_file"]
        elif mode == "train":
            file_name = self.data_config["train_file"]
        else:
            raise ValueError("mode는 train 또는 test 중 하나여야 합니다.")

        return Dataset.from_pandas(self.load_dataset(pd.read_csv(self.data_path + "/" + file_name)))

    def make_user_messages(self, row: dict) -> str:
        """데이터셋의 각 row를 이용하여 사용자 프롬프트를 생성합니다.

        Args:
            row (dict): "paragraph", "question", "choices", "question_plus" 필드를 포함해야 합니다.

        Returns:
            str: 사용자 프롬프트를 반환합니다.
        """
        choice_string = "\n".join(
            [f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choice_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choice_string,
            )

        return user_message

    def make_chat_message(self, row: dict, user_message: str) -> dict:
        """데이터셋의 각 row와 그 row로부터 얻은 사용자 프롬프트를 이용하여 챗봇 프롬프트를 포함하는 딕셔너리를 생성합니다.
        상속받는 클래스에서 이 메소드를 오버라이드할 것.

        Args:
            row (dict): "id", "answer" 필드를 포함해야 합니다.
            user_message (str): 사용자 프롬프트 

        Returns:
            dict: 챗봇 프롬프트를 포함한 딕셔너리. Dataset으로 변환됩니다.
        """
        return {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "label": row["answer"],
        }

    def _make_chat_message(self, row: dict, user_message: str, mode="train") -> dict:
        """데이터셋의 각 row와 그 row로부터 얻은 사용자 프롬프트를 이용하여 챗봇 프롬프트를 포함하는 딕셔너리를 생성합니다.
        데이터셋 모드에 따라 챗봇 프롬프트에 정답을 포함할지 여부를 결정합니다.

        Args:
            row (dict): "id", "answer", "choices" 필드를 포함해야 합니다.
            user_message (str): 사용자 프롬프트 
            mode (str): 데이터셋 모드 (train, test 중 하나)

        Returns:
            dict: 챗봇 프롬프트를 포함한 딕셔너리. Dataset으로 변환됩니다.
        """
        chat_message = self.make_chat_message(row, user_message)
        if mode == "train":
            chat_message["messages"].append(
                {"role": "assistant", "content": f"{chat_message['label']}"})
        elif mode == "test":
            chat_message["len_choices"] = len(row["choices"])
        else:
            raise ValueError("mode는 train 또는 test 중 하나여야 합니다.")

        return chat_message

    def process_dataset(self, dataset: Dataset, mode="train") -> Dataset:
        """입력으로 들어온 dataset을 이용하여 입력 프롬프트와 그에 대한 정답을 생성합니다.

        Args:
            dataset (Dataset): 입력으로 들어온 dataset. 아래 필드를 포함하고 있어야 합니다.
            (id-식별자, 지문-paragraph, 질문-question, 선지-choices, 보기-question_plus,정답-answer)
            mode (str): 데이터셋 모드 (train, test 중 하나)

        Returns:
            Dataset: 아래 필드를 포함하고 있는 dataset을 반환합니다.
            (id-식별자, messages-프롬프트를 위한 message, label-정답)
        """
        processed_dataset = []
        for i in range(len(dataset)):
            user_message = self.make_user_messages(dataset[i])

            if self.data_config.get("use_4-choices_prompt", False):
                if len(dataset[i]["choices"]) == 4:
                    user_message = user_message.replace(
                        "1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.", "1, 2, 3, 4 중에 하나를 정답으로 고르세요.")

            # chat message 형식으로 변환
            processed_dataset.append(
                self._make_chat_message(dataset[i], user_message, mode=mode)
            )

        return Dataset.from_pandas(pd.DataFrame(processed_dataset))

    def compute_metrics(self, evaluation_result):
        int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        acc_metric = evaluate.load("accuracy")

        logits, labels = evaluation_result
        tokenizer = self.manager.tokenizer
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split(
            "<end_of_turn")[0].strip(), labels))
        labels = list(map(lambda x: int_output_map[x], labels))

        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        acc = acc_metric.compute(predictions=predictions, references=labels)

        return acc

    def report_metrics(self, metrics):
        # print("-" * 30)
        # print("Evaluation Accuracy : {:.4f}".format(metrics["eval_accuracy"]))
        # print("-" * 30)
        pass

    def get_train_and_valid_df(self, eval_dataset: Optional[Dataset]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dev_file = self.data_config.get("dev_file", None)
        if dev_file is not None:
            # dev_file을 직접 지정한 경우, 이 경우에는 이 함수를 호출하지 않는 것을 추천합니다.
            return (
                pd.read_csv(self.data_path + "/" +
                            self.data_config["train_file"]),
                pd.read_csv(self.data_path + "/" + dev_file)
            )
        else:  # splited
            all_df = pd.read_csv(self.data_path + "/" +
                                 self.data_config["train_file"])
            valid_df_ids = set(
                eval_dataset["id"]) if eval_dataset is not None else set()
            train_df = all_df[~all_df["id"].isin(valid_df_ids)]
            eval_df = all_df[all_df["id"].isin(valid_df_ids)]
            return train_df, eval_df

    def train(self):
        # train task
        dataset = self._load_dataset()

        if self.manager.model is None:
            self.manager.set_model()
        if self.manager.tokenizer is None:
            self.manager.set_tokenizer()

        processed_dataset = self.process_dataset(dataset)

        tokenized_dataset = processed_dataset.map(
            self.manager.apply_chat_template_and_tokenize,
            remove_columns=list(processed_dataset.features),
            batched=True,
            num_proc=self.data_config.get("tokenizer_num_procs", 1),
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        # 데이터 분리
        # 데이터 길이 default = 1024
        filter_len = self.data_config.get("filtering_input_ids_length", 1024)
        if filter_len > 0:
            tokenized_dataset = tokenized_dataset.filter(
                lambda x: len(x["input_ids"]) <= filter_len)

        # 데이터셋 분리 (test_size 설정과 dev_file 설정 고려)
        test_size = float(self.data_config.get("test_size", 0.1))
        dev_file = self.data_config.get("dev_file", None)
        if test_size > 0 and dev_file is None:
            tokenized_dataset = tokenized_dataset.train_test_split(
                test_size=test_size, seed=42)

            train_dataset = tokenized_dataset['train']
            eval_dataset = tokenized_dataset['test']
        elif dev_file is not None:
            train_dataset = tokenized_dataset

            processed_dev_dataset = self.process_dataset(
                self._load_dataset(mode="dev"))
            tokenized_dev_dataset = processed_dev_dataset.map(
                self.manager.apply_chat_template_and_tokenize,
                remove_columns=list(processed_dataset.features),
                batched=True,
                num_proc=self.data_config.get("tokenizer_num_procs", 1),
                load_from_cache_file=True,
                desc="Tokenizing",
            )

            eval_dataset = tokenized_dev_dataset
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None

        # save train and eval dataset
        train_df, eval_df = self.get_train_and_valid_df(eval_dataset)
        if self.experiment_config.get("save_train_dataset", False):
            self.save_df(train_df, "train.csv")
        if self.experiment_config.get("save_eval_dataset", False):
            self.save_df(eval_df, "dev.csv")

        if self.manager.data_collator is None:
            self.manager.set_data_collator()

        if self.manager.trainer is None:
            self.manager.set_trainer(
                train_dataset, eval_dataset, self.compute_metrics, self.preprocess_logits_for_metrics)

        self.manager.train()

        # 훈련 종료시 evaluate를 진행할지 안할지 결정
        last_eval_strategy = self.experiment_config.get(
            "last_eval_strategy", "no")
        if (test_size > 0 or dev_file is not None) and not last_eval_strategy == "no":
            if last_eval_strategy == "evaluate":
                final_metrics = self.manager.evaluate()
                self.report_metrics(final_metrics)
                self.save_json(final_metrics, "eval-metrics.json")
            elif last_eval_strategy == "inference":
                _, eval_df = self.get_train_and_valid_df(eval_dataset)
                processed_df = self.process_dataset(
                    self._load_dataset(df=eval_df), mode="test")
                output = self.do_inference(self.manager.model, processed_df)
                self.save_df(output, "eval-output.csv")

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

                target_logit_list = [
                    logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(target_logit_list, dtype=torch.float32)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
                infer_results.append({"id": _id, "answer": predict_value})

        return pd.DataFrame(infer_results)

    def inference(self) -> pd.DataFrame:
        """테스트 데이터셋에 대한 추론을 수행합니다.

        Returns:
            pd.DataFrame: 추론 결과를 반환합니다.
        """
        dataset = self._load_dataset(mode="test")
        if self.manager.model is None:
            self.manager.set_model(AutoModel=AutoPeftModelForCausalLM)
        elif self.manager.trainer.state.best_model_checkpoint is not None:
            original_model_name = self.manager.model_name_or_checkpoint
            self.manager.model_name_or_checkpoint = self.manager.trainer.state.best_model_checkpoint
            self.manager.set_model(AutoModel=AutoPeftModelForCausalLM)
            self.manager.model_name_or_checkpoint = original_model_name
        elif self.manager.trainer is not None:
            self.manager.model = self.manager.trainer.model

        if self.manager.tokenizer is None:
            self.manager.set_tokenizer()

        test_dataset = self.process_dataset(dataset, mode="test")
        output = self.do_inference(self.manager.model, test_dataset)
        self.save_df(output, "output.csv")
        print("Successfully saved the output csv file!")

        return output
