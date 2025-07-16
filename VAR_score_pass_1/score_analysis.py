import numpy as np
import os
from tqdm import tqdm
from collections import Counter
import sys
import pandas as pd
from typing import Iterable, Union, Any
from pathlib import Path
import json


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


dataset_list = "VAR_amc23,amc23,VAR_aime24,aime24"

dataset_list = dataset_list.split(",")
model_name_list = [
    ("DeepSeek-R1-0528", "none"),
    ("SEED-THINK-v1.6", "none"),
    ("Qwen3-235B-A22B", "none"),
    ("OpenAI-o4-mini-high", "none"),
    ]


df_data = []

for dataset in dataset_list:
    for (model_name, prompt_type) in model_name_list:

        pre_path = f"VAR_score_pass_1/{model_name}/{dataset}"
        pattern = f"{prompt_type}_-1_seed0_t0.0_s0_e-1.jsonl"
        pass1_sample_filepath = [val for val in os.listdir(pre_path) if val.endswith(pattern)]
        assert len(pass1_sample_filepath) == 1, f"{model_name}: {pass1_sample_filepath}"
        pass1_sample_filepath = pass1_sample_filepath[0]
        pass1_sample_filepath = os.path.join(pre_path, pass1_sample_filepath)
        pass1_sample_list = sorted(list(load_jsonl(pass1_sample_filepath)), key=lambda x: x["idx"])  
        

        data = {}
        for pass1_sample in pass1_sample_list:
            q_id = str(pass1_sample["id"]).split("_")[0]
            data.setdefault(q_id, []).append(pass1_sample["score"][0])
        num_refined_samples = sum(len(val)>1 for val in data.values())
        num_samples = len(data)
        
        data_score = {}
        avg_score = []
        for q_id, scores in data.items():
            data_score[q_id] = int(np.all(scores))
            avg_score.append(np.mean(np.array(scores)))
        
        avg_score = np.mean(avg_score)
        
        tmp_data = [model_name, dataset, prompt_type, np.mean(list(data_score.values())), avg_score, num_refined_samples, num_samples]
        df_data.append(tmp_data)

df = pd.DataFrame(df_data, columns=["model_name", "dataset", "prompt_type", "data_score", "avg_score", "num_refined_samples", "num_samples"])
df["data_score"] = df["data_score"].apply(lambda x: round(x * 100, 2))
df["avg_score"] = df["avg_score"].apply(lambda x: round(x * 100, 2))
print("============ Detail infos ============")
print(df)


pre_col = "data_score"
df = df[["model_name", "dataset", pre_col]]
df_pivot = df.pivot(index="model_name", columns="dataset", values=pre_col)
show_cols = []
for dataset in dataset_list:
    if dataset.startswith("VAR_"):
        comp_dataset = dataset[4:]
        df_pivot["Diff " + comp_dataset] = (df_pivot[dataset] - df_pivot[comp_dataset]) / df_pivot[comp_dataset] * 100
        show_cols.extend([comp_dataset, dataset, "Diff " + comp_dataset])
df_pivot[show_cols] = df_pivot[show_cols].applymap(lambda x: round(x, 1))

print("============ Results Table ============")
print(df_pivot[show_cols])