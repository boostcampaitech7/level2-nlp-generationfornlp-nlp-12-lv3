from datasets import load_dataset
import pandas as pd
import re

dataset = load_dataset("EunsuKim/CLIcK")

dataframe = []

for data in dataset["train"]:
    row = {"question_id": data["id"]}

    row["paragraph"] = data["paragraph"]

    if not data["paragraph"]:
        question_pattern = re.compile(r"(?:<보기>의 개요를 수정\?보완할[^?]+\?)|(?:[^.!?\n]+?(?<!보완)\?)|(?:[^.!?\n]+?(?:고르시오|설명하시오|예를\s+들어주세요|고르십시오)\.?)")
        question = re.findall(question_pattern, data["question"])[0]
        row["paragraph"], row["question_plus"] = re.split(question_pattern, data["question"], 1)
        data["question"] = question

    data["answer"] = data["choices"].index(data["answer"]) + 1

    row["problems"] = f"{{'question': '{data['question']}', 'choices': {data['choices']}, 'answer': {data['answer']}}}"

    if not "question_plus" in row:
        row["question_plus"] = ""

    ordered_row = {
        "question_id": row["question_id"],
        "paragraph": row["paragraph"],
        "problems": row["problems"],
        "question_plus": row["question_plus"]
    }

    dataframe.append(ordered_row)

df = pd.DataFrame(dataframe)
df.to_csv(f"data/click_data.csv", index=False, encoding='utf-8-sig')