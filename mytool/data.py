import os

import numpy as np
from datasets import load_dataset

from llama.tokenizer import Tokenizer
from functools import wraps

def get_c4(split, proxy: bool = False):
    if proxy:
        os.environ["HTTP_PROXY"] = "http://localhost:7890"
        os.environ["HTTPS_PROXY"] = "http://localhost:7890"
    else:
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        del os.environ["HTTP_PROXY"]
        del os.environ["HTTPS_PROXY"]

    if split == "train":
        data = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    else:
        raise ValueError("Invalid split: {}".format(split))

    return data


def get_data(
    name,
    split: str,
    first_n: int = None,
    tokenize: bool = True,
    tokenizer_path: str = "facebook/tokenizer.model",
    bos: bool = True,
    eos: bool = False,
    **kwargs
):
    if name == "c4":
        data = get_c4(split, **kwargs)
    else:
        raise NotImplementedError

    if not tokenize:
        return data[:first_n]

    tokenizer = Tokenizer(tokenizer_path)

    print(data)

    tokens = [tokenizer.encode(x, bos=bos, eos=eos) for x in data["text"][:first_n]]

    return tokens



def _test():
    data = get_data("c4", "train", tokenize=False, proxy=True)
    print(data["text"][0])
    assert isinstance(data["text"][0], str)

    data = get_data("c4", "validation", tokenize=True, proxy=True, bos=True, eos=True)
    print(data[0])
    assert isinstance(data[0], list)
    assert isinstance(data[0][0], int)
    assert data[0][0] == 1
    assert data[0][0] == 2


if __name__ == "__main__":
    _test()
