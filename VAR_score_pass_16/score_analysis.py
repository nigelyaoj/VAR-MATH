import numpy as np
import os
from torch import mode
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


dataset_list = "VAR_amc23,amc23"
# dataset_list = "VAR_aime24,aime24"
# dataset_list = "VAR_aime25,aime25"

N_samples = 16
N_boostrap = 1000
agg_mode = "mean"  # all or mean

dataset_list = dataset_list.split(",")
model_name_list = [
    ("Qwen/Qwen2.5-MATH-7B", "qwen25-math-cot"),
    ("PRIME-RL/Eurus-2-7B-PRIME", "qwen25-math-cot"),
    ("hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo", "qwen25-math-cot"),
    ("sail/Qwen2.5-Math-7B-Oat-Zero", "qwen25-math-cot"),
    ("Skywork/Skywork-OR1-Math-7B", "deepseek-r1"),
    ("qihoo360/Light-R1-7B-DS", "deepseek-r1"),
    ]

model_name_list += [
    ("BytedTsinghua-SIA/DAPO-Qwen-32B", "qwen2.5-32B"),
    ("Kwaipilot/SRPO-Qwen-32B", "SRPO"),
    ("Qwen/Qwen2.5-32B", "qwen2.5-32B"), 
]

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
        for idx in range(N_boostrap):
            data = {}
            for pass1_sample in pass1_sample_list:
                q_id = str(pass1_sample["id"]).split("_")[0]
                if "VAR_" in dataset:
                    random_idx = np.random.choice(N_samples, 1)[0]
                else:
                    random_idx = idx % N_samples
                data.setdefault(q_id, []).append(pass1_sample["score"][random_idx])
            
            data_score = {}
            for q_id, scores in data.items():
                if agg_mode == "all":
                    data_score[q_id] = int(np.all(scores))
                elif agg_mode == "mean":
                    data_score[q_id] = np.mean(scores)
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
df["data_score_std"] = df["data_score_std"].apply(lambda x: round(x * 100, 2))
print("============ Detail infos ============")
print(df)

'''
############################ plot the results ############################
import matplotlib.pyplot as plt
# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create subplots for the dataset pairs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Filter data for amc23 and VAR-amc23
amc23_data = df[df['dataset'] == 'amc23']['data_score_std']
var_amc23_data = df[df['dataset'] == 'VAR_amc23']['data_score_std']

# Filter data for aime24 and VAR-aime24
aime24_data = df[df['dataset'] == 'aime24']['data_score_std']
var_aime24_data = df[df['dataset'] == 'VAR_aime24']['data_score_std']

# Filter data for aime25 and VAR-aime25
aime25_data = df[df['dataset'] == 'aime25']['data_score_std']
var_aime25_data = df[df['dataset'] == 'VAR_aime25']['data_score_std']

# Combine data for boxplot
combined_data = pd.DataFrame({
    'Standard Deviation of Score': pd.concat([
        amc23_data, var_amc23_data, 
        aime24_data, var_aime24_data, 
        aime25_data, var_aime25_data
    ]),
    'Dataset': (['AMC23'] * len(amc23_data) + 
                ['VAR-AMC23'] * len(var_amc23_data) + 
                ['AIME24'] * len(aime24_data) + 
                ['VAR-AIME24'] * len(var_aime24_data) + 
                ['AIME25'] * len(aime25_data) + 
                ['VAR-AIME25'] * len(var_aime25_data))
})

# Create a single boxplot
sns.boxplot(data=combined_data, x='Dataset', y='Standard Deviation of Score', palette='husl', ax=ax1)
ax1.set_title('Standard Deviation of Score Across Datasets')
ax1.set_xlabel('Dataset')
ax1.set_ylabel('Standard Deviation of Score')
ax1.grid(True, alpha=0.3)

# Hide the second subplot (ax2) since we are using only one plot
fig.delaxes(ax2)

plt.tight_layout()
plt.savefig('data_score_std_density_distribution.png', dpi=300, bbox_inches='tight')
'''


'''
############################ Calculate variance comparison statistics ############################
# Calculate proportion for AMC23
amc23_higher_count = 0
total_models = len(model_name_list)

for model_name, _ in model_name_list:
    amc23_std = df[(df['dataset'] == 'amc23') & (df['model_name'] == model_name)]['data_score_std'].iloc[0]
    var_amc23_std = df[(df['dataset'] == 'VAR_amc23') & (df['model_name'] == model_name)]['data_score_std'].iloc[0]
    if amc23_std > var_amc23_std:
        amc23_higher_count += 1

amc23_proportion = amc23_higher_count / total_models

# Calculate proportion for AIME24
aime24_higher_count = 0
for model_name, _ in model_name_list:
    aime24_std = df[(df['dataset'] == 'aime24') & (df['model_name'] == model_name)]['data_score_std'].iloc[0]
    var_aime24_std = df[(df['dataset'] == 'VAR_aime24') & (df['model_name'] == model_name)]['data_score_std'].iloc[0]
    if aime24_std > var_aime24_std:
        aime24_higher_count += 1

aime24_proportion = aime24_higher_count / total_models

# Create and display variance comparison table
variance_comparison_data = {
    'Dataset Comparison': ['AMC23 vs VAR-AMC23', 'AIME24 vs VAR-AIME24'],
    'Models with Higher Original Variance': [amc23_higher_count, aime24_higher_count],
    'Total Models': [total_models, total_models],
    'Proportion (%)': [round(amc23_proportion * 100, 1), round(aime24_proportion * 100, 1)]
}

variance_df = pd.DataFrame(variance_comparison_data)
print("============ Variance Comparison Statistics ============")
print(variance_df)
'''

########################### Results Table ###########################
df = df[["model_name", "dataset", "data_score", "data_score_std"]]
df_pivot = df.pivot(index="model_name", columns="dataset", values=["data_score", "data_score_std"])

# Flatten the multi-level columns
df_pivot.columns = [f"{col[1]} ({col[0]})" for col in df_pivot.columns]

show_cols = []
for dataset in dataset_list:
    if dataset.startswith("VAR_"):
        comp_dataset = dataset[4:]
        df_pivot[f"Diff {comp_dataset}"] = (
            (df_pivot[f"{dataset} (data_score)"] - df_pivot[f"{comp_dataset} (data_score)"]) / 
            df_pivot[f"{comp_dataset} (data_score)"] * 100
        )
        show_cols.extend([
            f"{comp_dataset} (data_score)", f"{comp_dataset} (data_score_std)",
            f"{dataset} (data_score)", f"{dataset} (data_score_std)", 
            f"Diff {comp_dataset}"
        ])
df_pivot[show_cols] = df_pivot[show_cols].applymap(lambda x: round(x, 1) if isinstance(x, (int, float)) else x)

print("============ Results Table ============")
print(df_pivot[show_cols])
print(df_pivot[show_cols].mean())