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
        model=model,  # qwen3-235b-a22b
        messages=messages,
        # enable_thinking 
        extra_body={"enable_thinking": True},
        stream=True,
        # stream_options={
        #     "include_usage": True
        # },
    )

    reasoning_content = ""  
    answer_content = ""  

    for chunk in completion:
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content

        if hasattr(delta, "content") and delta.content:
            answer_content += delta.content
    
    rep_dict = {}
    rep_dict["response"] = answer_content
    rep_dict["reasoning"] = reasoning_content

    return rep_dict
    

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_Qwen3_API_outputs(model_name, prompts):

    if model_name == "Qwen3-235B-A22B":
        model = "qwen3-235b-a22b"
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


if __name__ == "__main__":
    # Example usage
    model_name = "Qwen3-235B-A22B"
    prompts = ["What is the capital of France?", "Explain the theory of relativity.", "2*399=?"]
    outputs = get_Qwen3_API_outputs(model_name, prompts)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}: {output}")