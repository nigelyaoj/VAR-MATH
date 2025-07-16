import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


PROMPT_TEMPLATES = {
    "none": ("{input}", "{output}", "\n\n"),
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "deepseek-r1": (
        "<｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜><think>\n",
        "{output}",
        "\n\n",
    ),
    "qwen2.5-32B": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\n<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "SRPO": (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        "User:{input}\n",
        "Assistant: <think>",
        "{output}",
        "\n\n",
    ),    
    "llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 03 Jun 2025\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "{output}",
        "\n\n",
    ),
    "spurious_rewards_prompt": (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n<|user|>\n{input}\n<|assistant|>\n<think>",
        "{output}",
        "\n\n",
    ),
    "div-cot": (
        "<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant should 1) Identify core concepts and required formulas. 2) Break down solutions into logical, numbered steps. 3) Verify results using alternative methods or substitutions. Put your final answer within \\boxed{{}}<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "deepseek-longcot": (
    "<|im_start|>system\n"
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n",
    "{output}",
    "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
    "o1_cot": (
        '[Round 0] USER:\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}. ASSISTANT:\n',
        "{output}",
        "\n\n"
    )
}


def construct_prompt(example, data_name, args):
   
    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    
    context = input_template.format(input=example["question"])
    full_prompt = context

    return full_prompt.strip(" ")  # important!