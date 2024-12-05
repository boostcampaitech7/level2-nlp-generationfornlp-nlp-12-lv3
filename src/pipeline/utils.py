import json
from git import Repo
from datasets import Dataset
from box import Box
import pandas as pd
from transformers import StoppingCriteria

def load_config(config_path: str):
    config_path = config_path + ".json"

    with open(config_path, 'r') as f:
        config = json.load(f)
        config = Box(config)

    return config

def load_last_commit(repo_path="."):
    try:
        repo = Repo(repo_path)
        return repo.head.commit.hexsha
    except Exception as e:
        print("Error while fetching the last commit hash:", e)
        return None
    
class StopOnText(StoppingCriteria):
    def __init__(self, tokenizer, stop_text):
        self.tokenizer = tokenizer
        self.stop_text = stop_text
        self.stop_ids = tokenizer.encode(stop_text, add_special_tokens=False)

    def __call__(self, input_ids, scores):
        # 마지막 생성된 토큰들이 stop_ids와 일치하는지 확인
        generated_text = self.tokenizer.decode(input_ids[0])
        return generated_text.endswith(self.stop_text)

class StopOnAnswer(StoppingCriteria):
    def __init__(self, tokenizer, stop_text):
        self.tokenizer = tokenizer
        self.answer_template = stop_text

    def __call__(self, input_ids, scores):
        # 마지막 생성된 토큰들이 stop_ids와 일치하는지 확인
        generated_text = self.tokenizer.decode(input_ids[0])
        for choice in [1, 2, 3, 4, 5]:
            if self.answer_template.format(choice) in generated_text:
                return True
        return False