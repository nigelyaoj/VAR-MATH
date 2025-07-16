import random
import os
import argparse
import time
from time import sleep

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from openai import OpenAI

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="amc23", type=str)
    parser.add_argument("--data_dir", default="VAR_data", type=str)
    parser.add_argument("--model_name_or_path", default="DeepSeek-R1-0528", type=str)
    parser.add_argument("--output_dir", default="VAR_score_pass_1/DeepSeek-R1-0528", type=str)
    parser.add_argument("--prompt_type", default="none", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=5, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=3000, type=int)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    args.save_outputs = True
    return args

def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples if len(sample["code"][0])>0} # modified for api
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    
    data_list = args.data_names.split(",")
   
    # infer & eval
    results = []
    for data_name in data_list:
        results.append(main(args.model_name_or_path, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(model_name_or_path, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "id": example["id"],
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]

    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "deepseek-longcot" in args.prompt_type:
        stop_words.append("</answer>")

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call): 
        print("-" * 20, "Epoch", epoch, len(remain_prompts))
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break
     
        prompts = [item[1] for item in current_prompts]
        
        if model_name_or_path in ["DeepSeek-R1-0528", "DeepSeek-V3-0324"]:
            from api_utils.deepseek_api import get_DS_API_outputs
            outputs = get_DS_API_outputs(
                model_name_or_path,
                prompts=prompts
            )
        elif model_name_or_path in ["Qwen3-235B-A22B"]:
            from api_utils.qwq_api import get_Qwen3_API_outputs
            outputs = get_Qwen3_API_outputs(
                model_name_or_path,
                prompts=prompts
            )
        elif model_name_or_path in ["SEED-THINK-v1.6"]:
            from api_utils.seed_think_api import get_seed_think_API_outputs
            outputs = get_seed_think_API_outputs(
                model_name_or_path,
                prompts=prompts
            )
        elif model_name_or_path in ["Gemini-2.5-pro", "OpenAI-o3-mini", "OpenAI-o4-mini-high"]:
            from api_utils.api import get_API_outputs
            outputs = get_API_outputs(
                model_name_or_path,
                prompts=prompts
            )
        else:
            raise ValueError(f"Unknown API, model name: {model_name_or_path}")

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            end_prompts.append((i, query))

        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            # assert False
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = [] # record completion
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        new_preds = []
        for val in preds:
            if val is None:
                new_preds.append("None")
                print("None")
            else:
                new_preds.append(val)
        preds = new_preds
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
