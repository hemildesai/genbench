import multiprocessing as mp
import os
import platform
import time
from typing import Any, cast

import pandas as pd
import tiktoken
from datasets import load_dataset

from genbench.llm.utils import TokenizerRecord

NUM_THREADS = [1, 4, 8, 16, 32]


def get_documents() -> list[str]:
    dataset = load_dataset("rotten_tomatoes")
    docs: list[dict[str, str]] = (
        [d for d in dataset["train"]]  # type: ignore
        + [d for d in dataset["validation"]]  # type: ignore
        + [d for d in dataset["test"]]  # type: ignore
    )
    doc_text = list(map(lambda x: x["text"], docs))
    return [" \n".join(doc_text)] * 100


def run_tokenizer_benchmark() -> list[TokenizerRecord]:
    docs = get_documents()
    records = []
    num_cores = mp.cpu_count()
    for num_threads in sorted(NUM_THREADS):
        if num_threads > num_cores:
            break
        records.extend(
            benchmark_batch(docs, num_threads, platform.processor(), num_cores)
        )

    return records


def get_tokenizer_benchmark_df() -> pd.DataFrame:
    records = run_tokenizer_benchmark()
    return pd.DataFrame([r.__dict__ for r in records])


# Adopted from https://github.com/openai/tiktoken/blob/main/scripts/benchmark.py
def benchmark_batch(
    documents: list[str], num_threads: int, processor: str, num_cores: int
) -> list[TokenizerRecord]:
    original_threads = os.environ.get("RAYON_RS_NUM_THREADS", None)
    records = []
    num_bytes = sum(map(len, map(str.encode, documents)))
    print(f"num_threads: {num_threads}, num_bytes: {num_bytes}")

    os.environ["RAYON_RS_NUM_THREADS"] = str(num_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("warmup")

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=num_threads)
    end = time.perf_counter_ns()
    print(f"tiktoken \t{num_bytes / (end - start) * 1e9} bytes / s")
    records.append(
        TokenizerRecord(
            tokenizer="tiktoken",
            num_threads=num_threads,
            num_bytes=num_bytes,
            time=(end - start) * 1e6,
            throughput=num_bytes / (end - start) * 1e9,
            processor=processor,
            cores=num_cores,
        )
    )

    import transformers

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("warmup")

    start = time.perf_counter_ns()
    hf_enc(documents)
    end = time.perf_counter_ns()
    print(f"huggingface \t{num_bytes / (end - start) * 1e9} bytes / s")
    records.append(
        TokenizerRecord(
            tokenizer="huggingface",
            num_threads=num_threads,
            num_bytes=num_bytes,
            time=(end - start) * 1e6,
            throughput=num_bytes / (end - start) * 1e9,
            processor=processor,
            cores=num_cores,
        )
    )

    if original_threads:
        os.environ["RAYON_RS_NUM_THREADS"] = original_threads

    return records
