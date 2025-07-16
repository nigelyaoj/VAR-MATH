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

    completion = client.chat.completions.create(
    extra_body={},
    model=model, 
    messages=messages
    )
    
    rep_dict = {}
    rep_dict["response"] = completion.choices[0].message.content

    return rep_dict


client = OpenAI(
  base_url="XXXX", # fill the base URL for your API
  api_key=os.getenv("API_KEY"),
)


def get_API_outputs(model_name, prompts):

    if model_name == "Gemini-2.5-pro":
        model = "google/gemini-2.5-pro"
    elif model_name == "OpenAI-o4-mini-high":
        model = "openai/o4-mini-high"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    task_args = []
    for idx, prompt in enumerate(prompts):
        message = [{"role": "user", "content": prompt}]
        task_args.append((idx, message, model))

    outputs = process_map(process_task_with_retries, task_args, max_workers=5)
    outputs = sorted(outputs, key=lambda x: x["idx"])
    outputs = [item["response"]["response"] for item in outputs]

    return outputs


if __name__ == "__main__":
    # Example usage
    model_name = "Gemini-2.5-pro"
    prompts = ["2*399=?"]
    outputs = get_API_outputs(model_name, prompts)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}: {output}")