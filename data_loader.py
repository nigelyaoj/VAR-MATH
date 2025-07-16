import os
from utils import load_jsonl


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples
