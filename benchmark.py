import base64
import functools
import gzip
import json
import os
import random
import time
from typing import Any, cast

import tiktoken


def benchmark_batch(documents: list[str]) -> None:
    num_threads = int(os.environ["RAYON_NUM_THREADS"])
    num_threads = 8
    num_bytes = sum(map(len, map(str.encode, documents)))
    # print(f"num_threads: {num_threads}, num_bytes: {num_bytes}")

    # enc = tiktoken.get_encoding("gpt2")
    # enc.encode("warmup")

    # start = time.perf_counter_ns()
    # enc.encode_ordinary_batch(documents, num_threads=num_threads)
    # end = time.perf_counter_ns()
    # print(f"tiktoken \t{num_bytes / (end - start) * 1e9} bytes / s")

    import transformers

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("warmup")

    start = time.perf_counter_ns()
    hf_enc(documents)
    end = time.perf_counter_ns()
    print(f"huggingface \t{num_bytes / (end - start) * 1e9} bytes / s")



if __name__ == "__main__":
    doc_name = "wikitext-103/wiki.train.tokens"
    input_doc = ""

    with open(doc_name, "r+") as file1:
        # Reading from a file
        input_doc = file1.read()

    # print(input_doc)
    benchmark_batch(input_doc.split("\n"))
