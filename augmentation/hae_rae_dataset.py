from datasets import load_dataset, concatenate_datasets
import pandas as pd

ds = concatenate_datasets(
    [
    load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", "correct_definition_matching")['test'],
    load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", "date_understanding")["test"],
    load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", "general_knowledge")["test"],
    load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", "history")["test"],
    load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", "loan_words")["test"],
    load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", "rare_words")["test"],
    load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", "standard_nomenclature")["test"],
    ]
)

dataframe = []

for i, data in enumerate(ds):
    
    parts = data["query"].split("###")[:-2]
    
    question = parts[0].replace("\n", "").replace("다음", "")
    paragraph = "".join(parts[1:]).replace("\n", "")
    choices = data["options"].replace("\n", "") 
    answer = data["answer"].replace("(A)", "1").replace("(B)", "2").replace("(C)", "3").replace("(D)", "4").replace("(E)", "5")
    answer = int(answer)

    row = {
        "question_id": f"hae_rae_{i}",
        "paragraph": paragraph,
        "problems": f"{{'question': '{question}', 'choices': {choices}, 'answer': {answer}}}",
        "question_plus": ""
    }

    dataframe.append(row)

df = pd.DataFrame(dataframe)
df.to_csv(f"data/hae_rae_data.csv", index=False, encoding='utf-8-sig')
