import os
from time import sleep
from tqdm.contrib.concurrent import process_map
from openai import OpenAI


def process_task_with_retries(args, max_retries=20):
    idx, messages, model = args
    for attempt in range(max_retries):
        try:
            response = chat(messages, model)
            return {"idx": idx, "messages": messages, "response": response}
        except Exception as e:
            if attempt < max_retries - 1:
                sleep(1)  # Wait before retrying
            else:
                print(str(e))
                return {"idx": idx, "messages": messages, "response": {"response": str(e)}}


def chat(messages, model):

    # return "test"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )

    rep_dict = {}
    rep_dict["response"] = response.choices[0].message.content
    
    return rep_dict


client = OpenAI(api_key=os.environ["DS_API_KEY"], base_url="https://api.deepseek.com")

def get_DS_API_outputs(model_name, prompts):

    if model_name == "DeepSeek-R1-0528":
        model = "deepseek-reasoner"
    elif model_name == "DeepSeek-V3-0324":
        model = "deepseek-chat"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    task_args = []
    for idx, prompt in enumerate(prompts):
        message = [{"role": "user", "content": prompt}]
        task_args.append((idx, message, model))

    outputs = process_map(process_task_with_retries, task_args, max_workers=100)
    outputs = sorted(outputs, key=lambda x: x["idx"])
    outputs = [item["response"]["response"] for item in outputs]

    return outputs