import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from tqdm import tqdm
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts.enhancer import PROMPT_SYSTEM, PROMPT_USER

class CoTDatasetEnhancer:
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 temperature = 0.9,
                 ):
        self._set_secret_key()
        self.prompts = self._load_prompt()
        self.llm = ChatOpenAI(
            model_name = model_name,
            temperature = temperature
        )
    
    def _load_prompt(self):
        return {
            "system": PROMPT_SYSTEM,
            "user": PROMPT_USER
        }
        
    def _set_secret_key(self, secret_file: str = "secrets.json"):
        with open(secret_file, 'r') as f:
            secrets = json.load(f)
        os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
    
    def _process_row(self, index, row, chain):
        if row["reason"] is not None:
            return index, row["reason"], 0
        paragraph = row["paragraph"]
        problems = literal_eval(row["problems"])
        question = problems["question"]
        choices = problems["choices"]
        answer = problems["answer"]
        try:
            with get_openai_callback() as cb:
                jsn = chain.invoke({
                    "paragraph": paragraph,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                })
                return index, jsn["reason"], cb.total_cost
        except Exception as _:
            return index, None, 0
    
    def enhance(self, df):
        system_msg_prompt = SystemMessagePromptTemplate.from_template(
            self.prompts["system"]
        )
        
        parser = JsonOutputParser()
        
        human_msg_prompt = HumanMessagePromptTemplate.from_template(
            self.prompts["user"]
        )
        
        chat_prompt = ChatPromptTemplate(
            messages = [system_msg_prompt, human_msg_prompt],
        )
        
        chat_prompt = chat_prompt.partial(
            format_instance = parser.get_format_instructions()
        )
        
        chain = chat_prompt | self.llm | parser
        
        df = df.copy()
        if "reason" not in df.columns:
            df["reason"] = None
        total_cost = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._process_row, i, row, chain): i for i, row in df.iterrows()}
            progress = tqdm(as_completed(futures), total=len(futures), desc="Enhancing")
            for future in progress:
                i = futures[future]
                try:
                    index, reason, cost = future.result()
                    if reason:
                        df.loc[index, "reason"] = reason
                    total_cost += cost
                    progress.set_postfix({"total_cost": f"${total_cost:.2f}"})
                except Exception as _:
                    print(f"Error processing row {i}")  
        
        print(f"Total cost: ${total_cost:.2f}")        
        # if any column of reason is None
        if df["reason"].isnull().sum() > 0:
            print("Some rows are not enhanced! (maybe due to budget issue)")
        
        return df
    
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    
    enhancer = CoTDatasetEnhancer()
    df = enhancer.enhance(df)
    
    df.to_csv("data/train_enhanced.csv", index=False)
    print(df.head())