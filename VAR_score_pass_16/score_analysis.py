import numpy as np
import os
from tqdm import tqdm
from collections import Counter
import sys
import pandas as pd
from typing import Iterable, Union, Any
from pathlib import Path
import json
import seaborn as sns


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


dataset_list = "VAR_amc23,amc23,VAR_aime24,aime24"

N_samples = 16
dataset_list = dataset_list.split(",")
model_name_list = [
    ("Qwen/Qwen2.5-MATH-7B", "qwen25-math-cot"),
    ("PRIME-RL/Eurus-2-7B-PRIME", "qwen25-math-cot"),
    ("hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo", "qwen25-math-cot"),
    ("sail/Qwen2.5-Math-7B-Oat-Zero", "qwen25-math-cot"),
    ("Skywork/Skywork-OR1-Math-7B", "deepseek-r1"),
    ("qihoo360/Light-R1-7B-DS", "deepseek-r1"),
    ]

# model_name_list = [
#     ("BytedTsinghua-SIA/DAPO-Qwen-32B", "qwen2.5-32B"),
#     ("Kwaipilot/SRPO-Qwen-32B", "SRPO"),
#     ("Qwen/Qwen2.5-32B", "qwen2.5-32B"), 
# ]


df_data = []

for dataset in dataset_list:

    for (model_name, prompt_type) in model_name_list:

        pre_path = f"VAR_score_pass_16/{model_name}/{dataset}"
        
        pattern = f"{prompt_type}_-1_seed0_t0.6_s0_e-1.jsonl"
        pass1_sample_filepath = [val for val in os.listdir(pre_path) if val.endswith(pattern)]
        assert len(pass1_sample_filepath) == 1, f"{model_name}: {pass1_sample_filepath}"
        pass1_sample_filepath = pass1_sample_filepath[0]
        pass1_sample_filepath = os.path.join(pre_path, pass1_sample_filepath)
        pass1_sample_list = sorted(list(load_jsonl(pass1_sample_filepath)), key=lambda x: x["idx"])  
        
        data_score_list = []
        for idx in range(N_samples):
            data = {}
            for pass1_sample in pass1_sample_list:
                q_id = str(pass1_sample["id"]).split("_")[0]
                data.setdefault(q_id, []).append(pass1_sample["score"][idx])
            
            data_score = {}
            for q_id, scores in data.items():
                data_score[q_id] = int(np.all(scores))
                # data_score[q_id] = np.mean(scores)
            data_score_list.append(np.mean(list(data_score.values())))

            if idx == 0:
                num_refined_samples = sum(len(val)>1 for val in data.values())
                num_samples = len(data)
            else:
                assert num_refined_samples == sum(len(val)>1 for val in data.values()), "Number of refined samples should be the same across all samples."
                assert num_samples == len(data), "Number of samples should be the same across all samples."
        
        # solved_problem_ids = ",".join([q_id for q_id, score in data_score.items() if score == 1])
        tmp_data = [model_name, dataset, prompt_type, np.mean(data_score_list), np.std(data_score_list), num_refined_samples, num_samples]
        df_data.append(tmp_data)

df = pd.DataFrame(df_data, columns=["model_name", "dataset", "prompt_type", "data_score", "data_score_std", "num_refined_samples", "num_samples"])
df["data_score"] = df["data_score"].apply(lambda x: round(x * 100, 2))

############################ plot the results ############################
import matplotlib.pyplot as plt

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create subplots for the two dataset pairs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Filter data for amc23 and VAR-amc23
amc23_data = df[df['dataset'] == 'amc23']['data_score_std']
var_amc23_data = df[df['dataset'] == 'VAR_amc23']['data_score_std']

# Filter data for aime24 and VAR-aime24
aime24_data = df[df['dataset'] == 'aime24']['data_score_std']
var_aime24_data = df[df['dataset'] == 'VAR_aime24']['data_score_std']

# Plot amc23 vs VAR-amc23 density
sns.kdeplot(data=amc23_data, ax=ax1, color='#ff9999', label='AMC23', fill=True, alpha=0.75)
sns.kdeplot(data=var_amc23_data, ax=ax1, color='#82b0d3', label='VAR-AMC23', fill=True, alpha=0.75)
ax1.set_xlabel('Standard Deviation of Score')
ax1.set_ylabel('Density')
ax1.set_title('AMC23 vs VAR-AMC23')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot aime24 vs VAR-aime24 density
sns.kdeplot(data=aime24_data, ax=ax2, color='#ff9999', label='AIME24', fill=True, alpha=0.75)
sns.kdeplot(data=var_aime24_data, ax=ax2, color='#82b0d3', label='VAR-AIME24', fill=True, alpha=0.75)
ax2.set_xlabel('Standard Deviation of Score')
ax2.set_ylabel('Density')
ax2.set_title('AIME24 vs VAR-AIME24')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_score_std_density_distribution.png', dpi=300, bbox_inches='tight')

########################### Results Table ###########################
df = df[["model_name", "dataset", "data_score"]]
df_pivot = df.pivot(index="model_name", columns="dataset", values="data_score")
show_cols = []
for dataset in dataset_list:
    if dataset.startswith("VAR_"):
        comp_dataset = dataset[4:]
        df_pivot["Diff " + comp_dataset] = (df_pivot[dataset] - df_pivot[comp_dataset]) / df_pivot[comp_dataset] * 100
        show_cols.extend([comp_dataset, dataset, "Diff " + comp_dataset])
df_pivot[show_cols] = df_pivot[show_cols].applymap(lambda x: round(x, 1))

print("============ Results Table ============")
print(df_pivot[show_cols])
print(df_pivot[show_cols].mean())