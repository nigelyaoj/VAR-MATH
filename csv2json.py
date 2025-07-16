import pandas as pd
import json
import random
import numpy as np
import copy
from fractions import Fraction
import os
import math


var_question_gen_N = 5
seed = 42
    
random.seed(seed)
np.random.seed(seed)

def get_candidate_set(var_info, var_round_n):
    vars = var_info.strip().split("\n")
    candidate_set = {}
    
    for name_var in vars:
        name = name_var.split("=")[0].strip()
        var = name_var.split("=")[1].strip()
        if var.startswith("Random_linespace_"):
            # VAR_X=Random_linespace_[10,20,2],
            range_params = var.split("_")[-1].replace("[", "").replace("]", "").split(",")
            linespace = list(np.arange(int(range_params[0]), int(range_params[1]), float(range_params[2])))
            np.random.shuffle(linespace)
            candidate_set[name] = linespace[:var_question_gen_N]
        elif var.startswith("Random_Set_"):
            # VAR_X=Random_Set_{1/3,1/2,3/5},
            set_params = var.split("_")[-1]
            assert set_params.startswith("{") and set_params.endswith("}"), "Invalid format for Random_Set"
            set_params = set_params[1:-1].split(",")
            set_params = [str(x) for x in set_params]
            np.random.shuffle(set_params)
            candidate_set[name] = set_params[:var_question_gen_N]

        elif var.startswith("Fixed_Set_"):
            # VAR_X=Fixed_Set_{3,4,5,6,7}
            set_params = var.split("_")[-1]
            assert set_params.startswith("{") and set_params.endswith("}"), "Invalid format for Random_Set"
            set_params = set_params[1:-1].split(",")
            set_params = [str(x) for x in set_params]
            candidate_set[name] = set_params[:var_question_gen_N]

        elif var.startswith("Expression_"):
            # VAR_Y=Expression_(1+VAR_X) / (1-VAR_X)
            var = var.replace("Expression_", "")
            real_var_question_gen_N = len(list(candidate_set.values())[0])
            values = []
            for ii in range(real_var_question_gen_N):
                express = copy.deepcopy(var)
                for key, value in candidate_set.items():
                    express = express.replace(key, str(value[ii]))
                try:
                    # Currently, the answer of the experssion for variables must be an integer
                    values.append(np.round(eval(express),0)) 
                except Exception as e:
                    print(f"Error evaluating expression '{express}': {e}")
            candidate_set[name] = values
        
        else:
            raise ValueError(f"Unknown variable type: {var}")

        
        if var_round_n == 0:
            candidate_set[name] = [int(x) for x in candidate_set[name]]
        elif var_round_n > 0:
            candidate_set[name] = [np.round(float(x), var_round_n) for x in candidate_set[name]]
    
    return candidate_set


def get_answer(candidate_set, var_answer, round_n):
    # get answer corresponding to candidate set
    real_var_question_gen_N = len(list(candidate_set.values())[0])

    if var_answer.startswith("Fixed_Set_"):
        # VAR_answer=Fixed_Set_{1,2,3,4,5}
        set_params = var_answer.split("_")[-1]
        assert set_params.startswith("{") and set_params.endswith("}"), "Invalid format for Random_Set"
        set_params = set_params[1:-1].split(",")
        set_params = [float(x) for x in set_params]
        set_params = [float(np.round(x, round_n)) for x in set_params]

        return set_params[:real_var_question_gen_N]
    
    elif var_answer.startswith("Expression_"):
        var_answer = var_answer.replace("Expression_", "")
        answers = []
        for ii in range(real_var_question_gen_N):
            express = copy.deepcopy(var_answer)
            for key, value in candidate_set.items():
                express = express.replace(key, str(value[ii]))
            try:
                answers.append(float(np.round(eval(express), round_n)))  # TODO: consider sympify
            except Exception as e:
                print(f"Error evaluating expression '{express}': {e}")
    
        return answers

    else:
        raise ValueError(f"Unknown variable answer type: {var_answer}")



dataset_name_list = ["amc23", "aime24"]

### original data
for dataset_name in dataset_name_list:
    csv_file = f"VAR_data/VAR-Math - {dataset_name}.csv"
    save_path = f"VAR_data/{dataset_name}/test.jsonl"

    data = pd.read_csv(csv_file)
    ori_data = []

    for i in range(len(data)):
        tmp_data = {
            "id": str(data.iloc[i]["id"]),
            "question": data.iloc[i]["ori_question"],
            "answer": float(data.iloc[i]["ori_answer"])
        }
        ori_data.append(tmp_data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for data in ori_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


### VAR data
for dataset_name in dataset_name_list:
    csv_file = f"VAR_data/VAR-Math - {dataset_name}.csv"
    save_path = f"VAR_data/VAR_{dataset_name}/test.jsonl"

    data = pd.read_csv(csv_file)
    VAR_new_data = []

    for i in range(len(data)):

        ori_id = data.iloc[i]["id"]
        var_question = data.iloc[i]["VAR_question"]
        var_info = data.iloc[i]["VAR_info"]
        var_answer = data.iloc[i]["VAR_answer"]
        done = data.iloc[i]["done"]

        if done == 1:
            var_round_n = int(data.iloc[i]["VAR_round"])
            candidate_set = get_candidate_set(var_info, var_round_n)

            answer_round_n = int(data.iloc[i]["VAR_answer_round"])
            answers = get_answer(candidate_set, var_answer, answer_round_n)

            real_var_question_gen_N = len(list(candidate_set.values())[0])
            
            for j in range(real_var_question_gen_N):
                question = copy.deepcopy(var_question)
                for key, value in candidate_set.items():
                    question = question.replace(key, str(value[j]))
                new_data = {
                    "id": f"{ori_id}_{j}",
                    "question": question,
                    "answer": answers[j]
                }
                VAR_new_data.append(new_data)
        
        else:
            new_data = {
                "id": str(ori_id),
                "question": data.iloc[i]["ori_question"],
                "answer": float(data.iloc[i]["ori_answer"])
            }
            VAR_new_data.append(new_data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for data in VAR_new_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    df_debug = []
    for data in VAR_new_data:
        tmp_data = [data["id"], data["question"], data["answer"]]
        df_debug.append(tmp_data)
    df_debug = pd.DataFrame(df_debug, columns=["id", "question", "answer"])
    debug_save_path = save_path.replace("/test.jsonl", "_debug.csv")
    df_debug.to_csv(debug_save_path, index=False, encoding="utf-8-sig")


