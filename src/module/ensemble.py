import os
import math
import argparse
import collections
import numpy as np
import pandas as pd
from ast import literal_eval
from typing import Optional

val_suffix = '-val'

def compute_metric(pred_df, eval_file: str = "data/dev.csv"):
    df = pd.read_csv(eval_file)
    df['answer'] = df['problems'].apply(
        lambda x: int(literal_eval(x)['answer'])
    )
    # 정답과 예측값을 비교하여 정확도 계산
    # row마다 같은 id의 row를 찾아 정답과 예측값을 비교
    correct = 0
    total = 0
    for _, row in df.iterrows():
        if row['id'] not in pred_df['id'].values:
            continue
        pred = pred_df[pred_df['id'] == row['id']]['answer'].values[0]
        if row['answer'] == pred:
            correct += 1
        total += 1
    accuracy = correct / total
    return accuracy, correct, total
    
def voting(weights, predictions, args):
    ensembled = {}
    if args.vote == "soft":
        for id, probs in predictions.items():
            num_choices = len(probs[0])
            ensembled[id] = np.zeros(num_choices)
            for i, prob in enumerate(probs):
                ensembled[id] += prob * (weights[i] / sum(weights))
        
        output_df = pd.DataFrame({"id" : list(ensembled.keys())})
        for i in range(5):
            output_df[f"prob_{i+1}"] = [ensembled[id][i] if i < len(ensembled[id]) else None 
                                        for id in ensembled]
        output_df["answer"] = output_df.apply(lambda x: np.argmax(x[1:6]) + 1, axis=1)
        return output_df
    else:
        def custom_argmax(x):
            # x[1:6] 구간에서만 보기
            values = x[1:6] 
            # 최대값 찾기
            max_val = max(values)
            print(max_val)
            
            # 최대값을 가진 위치들 중 weight가 가장 큰 위치 찾기
            max_pos = max(
                [(i, w) for i, (v, w) in enumerate(zip(values, weights)) if v == max_val],
                key=lambda x: x[1]
            )[0]
            
            # 1을 더해서 반환 (1-based index)
            return max_pos + 1
        for id, probs in predictions.items():
            num_choices = len(probs[0])
            ensembled[id] = np.zeros(num_choices)
            for i, prob in enumerate(probs):
                ensembled[id] += prob * (1 if weights[i] > 0 else 0)
        
        output_df = pd.DataFrame({"id" : list(ensembled.keys())})
        for i in range(5):
            output_df[f"prob_{i+1}"] = [ensembled[id][i] if i < len(ensembled[id]) else None 
                                        for id in ensembled]
        output_df["answer"] = output_df.apply(lambda x: custom_argmax(x), axis=1)
        return output_df
    
def use_soft(row, temperature: float = 1.0):
    arr = []
    for i in range(1, 6):
        if math.isnan(row[f"logit_{i}"]):
            continue
        arr.append(row[f"logit_{i}"])
    arr = np.array(arr) / temperature
    exp = np.exp(arr - np.max(arr))
    prob = exp / np.sum(exp)
    return prob

def use_hard(row):
    # tie-breaking: choose the smallest index
    arr = np.zeros(5)
    answer = int(row["answer"])
    arr[answer - 1] = 1
    return arr
    
def ensemble(weights, file_names, mode, args):
    predictions = collections.defaultdict(list)
    
    for file_name in file_names:
        suffix = (val_suffix if mode == 'v' else "") + ".csv"
        df = pd.read_csv(f"{args.dirname}/{file_name}{suffix}")
        for i, row in df.iterrows():
            if args.vote == "soft":
                prob = use_soft(row, temperature=args.temperature)
            elif args.vote == "hard":
                prob = use_hard(row)
            else:
                print(f"Invalid voting method: {args.vote}")
                return
            predictions[row["id"]].append(prob)
        
    for id, preds in predictions.items():
        if len(preds) != len(weights):
            print(f"! ! ! Warning: '{id}' has {len(preds)} predictions, but {len(weights)} weights are given. ! ! !")
    
    output_df = voting(weights, predictions, args)
    
    if mode == 'v':
        acc, correct, total = compute_metric(output_df)
        print(f"Accuracy: {acc} ({correct}/{total})")
    else:
        acc = None
    
    return output_df, acc

def interactive(args):
    if args.dirname is None:
        dirname = input(
            "Enter the directory name where the prediction files are located (default 'data/ensemble'): "
        )
    else:
        dirname = args.dirname
    if dirname == "":
        dirname = "data/ensemble"
    args.dirname = dirname
    
    if args.mode is None:
        mode = input(
            "Enter the mode of the ensemble ('v(alidation)' or 'i(nference)', default: 'v'): "
        )
    else:
        mode = args.mode
    mode = 'v' if mode == "" else mode.lower()
    if mode == 'v':
        print(f"Validation mode is selected. Select only file names that end with '{val_suffix}'.")
    elif mode == 'i':
        print(f"Inference mode is selected. Select only file names that don't end with '{val_suffix}'.")
    else:
        print("Invalid mode.")
        return
        
    file_names = os.listdir(dirname)
    file_names = [a for a in file_names if a[-4:] == ".csv"]
    dev_file_names = [a for a in file_names if a[-(4+len(val_suffix)):] == f"{val_suffix}.csv"]
    inf_file_names = [a for a in file_names if a[-(4+len(val_suffix)):] != f"{val_suffix}.csv"]
    
    cutted_dev = [a[:-(len(val_suffix)+4)] for a in dev_file_names]
    cutted_inf = [a[:-4] for a in inf_file_names]

    cutted_dev.sort()
    cutted_inf.sort()
    
    if cutted_dev != cutted_inf:
        print("There are different files between validation and inference.")
        print("It may cause an error.")
    
    if mode == 'v':
        file_names = cutted_dev
    elif mode == 'i':
        file_names = cutted_inf
    else:
        assert False
    
    weights = [0.0] * len(file_names)
    
    page = 1
    while True:
        print(f"Page {page}")
        for i, file_name in enumerate(file_names[(page - 1) * 10: page * 10]):
            weight = weights[(page - 1) * 10 + i]
            suffix = val_suffix if mode == 'v' else ""
            print(f"{'v' if weight != 0 else ' '} {i}. {file_name}{suffix}", end="") 
            print(f"({weight})" if weight != 0 else "")
    
        choice = input("Enter the number of the file you want to check [0-9,a,s,d]: ")
        if choice == 's' or choice == "":
            break
        elif choice == 'a':
            page = max(1, page - 1)
            continue
        elif choice == 'd':
            page = min(len(file_names) // 10 + 1, page + 1)
            continue
        
        try:
            choice = int(choice)
            if choice < 0 or choice > 9:
                raise ValueError
        except ValueError:
            print("Please enter a number between 0 and 9.")
            continue
        
        select = (page - 1) * 10 + choice
        
        weight = input(f"Enter the weight of '{file_names[select]}' (default = 1.0): ")
        weight = float(weight) if weight != "" else 1.0
        weights[select] = weight
    
    print("Let's cook!")
    output, _ = ensemble(weights, file_names, mode, args)
    if args.output_name is None:
        output_name = input("Enter the name of the output file (default 'ensemble'): ")
    else:
        output_name = args.output_name
    if output_name == "":
        output_name = "ensemble"
    suffix = (val_suffix if mode == 'v' else "") + ".csv"
    if not os.path.exists(f"{dirname}/output"):
        os.mkdir(f"{dirname}/output")
    output.to_csv(f"{dirname}/output/{output_name}{suffix}", index=False)
    
    if mode == 'v':
        y_or_n = input("Do you want to ensemble the infence files with the same weights? [y/n]: ")
        if y_or_n == 'y':
            inf_output, _ = ensemble(weights, file_names, 'i', args)
            inf_output.to_csv(f"{dirname}/output/{output_name}.csv", index=False)

def loop_ensemble(args):
    if args.dirname is None:
        dirname = input(
            "Enter the directory name where the prediction files are located (default 'data/ensemble'): "
        )
    else:
        dirname = args.dirname
    if dirname == "":
        dirname = "data/ensemble"
    args.dirname = dirname
    
    file_names = os.listdir(dirname)
    file_names = [a for a in file_names if a[-4:] == ".csv"]
    dev_file_names = [a for a in file_names if a[-(4+len(val_suffix)):] == f"{val_suffix}.csv"]
    inf_file_names = [a for a in file_names if a[-(4+len(val_suffix)):] != f"{val_suffix}.csv"]
    
    cutted_dev = [a[:-(len(val_suffix)+4)] for a in dev_file_names]
    cutted_inf = [a[:-4] for a in inf_file_names]

    cutted_dev.sort()
    cutted_inf.sort()
    
    if cutted_dev != cutted_inf:
        print("Loop Ensemble: There are different files between validation and inference.")
        return
    
    file_names = cutted_dev
    acc = 0.0
    
    while acc < args.accuracy:
        weights = np.random.rand(len(file_names))
        output, acc = ensemble(weights, file_names, 'v', args)
    
    print("I found something!")
    for weight, file_name in zip(weights, file_names):
        print(f"{file_name}: {weight}")
    
    if args.output_name is None:
        output_name = input("Enter the name of the output file (default 'ensemble'): ")
    else:
        output_name = args.output_name
    if output_name == "":
        output_name = "ensemble"
    
    if not os.path.exists(f"{dirname}/output"):
        os.mkdir(f"{dirname}/output")
    output.to_csv(f"{dirname}/output/{output_name}{val_suffix}.csv", index=False)
    
    inf_output, _ = ensemble(weights, file_names, 'i', args)
    inf_output.to_csv(f"{dirname}/output/{output_name}.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirname", type=str, default=None, help="directory name")
    parser.add_argument("-o", "--output_name", type=str, default=None, help="output file name")
    parser.add_argument("-a", "--accuracy", type=float, default=None, help="target accuracy")
    parser.add_argument("-v", "--vote", type=str, default="soft", help="voting method")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="temperature for soft voting")
    parser.add_argument("-m", "--mode", type=str, default=None, help="mode of ensemble (v or i)")
    args = parser.parse_args()
    if args.accuracy is not None:
        loop_ensemble(args)
    else:
        interactive(args)

if __name__ == '__main__':
    main()