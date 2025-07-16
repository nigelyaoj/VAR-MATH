# 🔥 VAR-MATH: Probing True Mathematical Reasoning in Large Language Models via Symbolic Multi-Instance Benchmarks
> **Reassessing RL Enhancement Under Rigorous Testing**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

<div align="center">
  <img src="assets/eval.png" alt="Multi-Instance Verification" style="max-height: 300px; width: 95%;"/>
  <br><em>Multi-Instance Verification Framework</em>
</div>


🔎 Our findings reveal two paradigm-shifting insights about mathematical reasoning in LLMs:

1. **The RL Enhancement Illusion**  
   Our research fundamentally challenges the purported reasoning improvements from RL training: When evaluated through our variabilization framework, these models exhibit catastrophic performance collapses once stripped of benchmark-specific patterns, revealing that their perceived capabilities stem not from enhanced reasoning, but from specialized adaptation to fixed problem sets through shallow heuristics.

2. **The Contamination-Reasoning Double Bind**  
   We expose a critical dual pathology in popular models like Qwen base model that enables this illusion: First, severe data contamination artificially inflates benchmark performance while masking fundamental reasoning deficits; second, an inherent instability in symbolic processing prevents consistent generalization, creating models that can solve specific instantiations but fail to recognize equivalent mathematical structures across variations - what we term "surface competence without conceptual understanding."

Together, these findings reveal that current evaluation practices primarily measure benchmark-specific optimization rather than genuine mathematical reasoning—highlighting the need for a fundamental rethink in how we assess and develop mathematical AI systems.
To move forward, we must go beyond static benchmarks and account for both data contamination and symbolic inconsistency in evaluation. VAR-MATH offers a broadly applicable methodology—not limited to AMC23 or AIME24, but extendable to domains like program synthesis, formal logic, and decision-making.

Let’s build evaluations that reward structural generalization and behavioral consistency—not just memorized answers！


### Data Construction & Evaluation Pipeline
<div align="center">
  <img src="assets/process.png" alt="VAR-MATH Pipeline" style="max-height: 300px; width: 95%;"/>
  <br><em> The process consists of two stages: preprocessing and evaluation. During preprocessing, original math problems are symbolically abstracted by replacing fixed constants with variables, defining feasible sampling ranges, and expressing answers as parametric functions. These symbolic problem templates are stored in the VAR-MATH benchmark.In the evaluation stage, symbolic problems are instantiated with sampled values from the defined variable ranges, producing multiple concrete versions of the same underlying structure. Each instantiation is solved, and models must produce all correct answers to be considered successful. This strategy enforces reasoning consistency and mitigates contamination and evaluation instability.</em>
</div>

## 🔎 Key Findings
**Current RL methods show illusory gains on mathematical reasoning** - our variabilization approach reveals:

### 📉 RL-Trained 7B Models Show Severe Degradation
Revealing overfitting and inconsistent reasoning

| Model Type       | AMC23 | VAR-AMC23 | Δ       | AIME24 | VAR-AIME24 | Δ       |
|------------------|-------|-----------|---------|--------|------------|---------|
|**Qwen2.5-MATH-7B** | 36.9 | 2.5 | 🔻93.2% | 10.8 | 3.3 | 🔻69.3%|
|**Eurus-2-7B-PRIME** | 58.3 | 29.1 | 🔻50.1% | 15.8 | 4.4 |🔻72.3%|
|**Skywork-OR1-Math-7B** | 73.9 | 56.4 | 🔻23.7% | 41.5 | 24.4 |🔻41.2%|
|**Qwen-2.5-Math-7B-SimpleRL-Zoo** | 61.4 | 33.6 | 🔻45.3% | 23.8 | 8.3 | 🔻64.9%|
|**Light-R1-7B-DS** | 78.6 | 53.3 | 🔻32.2% | 40.8 | 24.4 | 🔻40.3%|
|**Qwen2.5-Math-7B-Oat-Zero** | 65.6 | 37.2 | 🔻43.3% | 34.0 | 12.9 | 🔻62.0%|

### ⚖️ 32B Models: Improved Accuracy ≠ Robust Generalization

| Model Type       | AMC23 | VAR-AMC23 | Δ       | AIME24 | VAR-AIME24 | Δ       |
|------------------|-------|-----------|---------|--------|------------|---------|
|**Qwen2.5-32B** | 33.4 | 2.5 | 🔻 92.5 % | 8.8 | 2.5 | 🔻 71.4 %|
|**DAPO-Qwen-32B** | 92.3 | 69.7 | 🔻 24.5 % | 51.7 | 30.6 | 🔻 40.7 %|
|**SRPO-Qwen-32B** | 86.7 | 51.2 | 🔻 40.9 % | 55.6 | 29.6 | 🔻 46.8 %|

### 🚀 Frontier Models: Greater Robustness, Persistent Gaps

| Model Type       | AMC23 | VAR-AMC23 | Δ       | AIME24 | VAR-AIME24 | Δ       |
|------------------|-------|-----------|---------|--------|------------|---------|
|**DeepSeek-R1-0528** | 100.0 | 100.0 | 🔻 0.0 % | 83.3 | 73.3 | 🔻 12.0 %|
|**OpenAI-o4-mini-high** | 100.0 | 87.5 | 🔻 12.5 % | 90.0 | 73.3 | 🔻 18.5 %|
|**Qwen3-235B-A22B** | 100.0 | 95.0 | 🔻 5.0 % | 83.3 | 70.0 | 🔻 16.0 %|
|**SEED-THINK-v1.6** | 100.0 | 97.5 | 🔻 2.5 % | 93.3 | 86.7 | 🔻 7.1 %|


## ⚙️ Setup
We provide the environment for evaluation, tested on Debian 6.1.106-3 with A6000 GPUs:
```bash
conda create -n VAR-MATH python==3.9.0
conda activate VAR-MATH
cd latex2sympy
pip install -e . 
cd ..
pip install -r requirements.txt
```

## ▶️ Run Evaluation

### Generate the VAR-Math data
```bash
python csv2json.py
```

We provide the original csv file for editing the questions.
We also provide the csv with suffix "_debug" for users to check the new generated questions.

### Evaluate
```bash
sh eval_local_7b.sh # for 7B-parameter models
sh eval_local_32b.sh # for 32B-parameter models
sh eval_api.sh # YOU SHOULD FILL THE API KEY AND THE BASE URL.
```

### Collect the results
```bash
python VAR_score_pass_1/score_analysis.py # for Large-scale models
python VAR_score_pass_16/score_analysis.py # for 7B- and 32B-parameter models
```

## Acknowledgments

We extend our sincere gratitude to:
- **AMC/AIME** for open-sourcing their mathematical problem collections
- The developers of evaluation frameworks:
  - [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness)
  - [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)
- The broader open-source community for making models and tools publicly available

## Citations
The paper is coming soon!

```bibtex
@misc{varmath2025,
  author = {Yao, Jian and Cheng, Ran and Tan, Kay Chen},
  title = {VAR-MATH: Probing True Mathematical Reasoning in Large Language Models via Symbolic Multi-Instance Benchmarks},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nigelyaoj/VAR-MATH}}
}
```