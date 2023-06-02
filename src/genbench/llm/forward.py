# Adapted from https://github.com/huggingface/optimum/blob/main/tests/benchmark/benchmark_bettertransformer.py
import itertools

import pandas as pd
import torch
from optimum.exporters import TasksManager
from torch.backends.cuda import SDPBackend
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from genbench.llm.utils import ModelForwardRecord
from genbench.optimizer import Optimizer
from genbench.utils import benchmark_transformers_with_memory, garbage_collect

set_seed(42)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore

BATCH_SIZES = [1, 8]
SEQ_LEN = [64, 128]
NUM_TOKENS = [128, 256]
SDP_BACKENDS = [
    SDPBackend.ERROR,  # This means no kernel specific context manager
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
]
DTYPES = [torch.float16, torch.bfloat16]
BETTERTRANSFORMER = [True]
COMPILE = [True, False]


def get_batch(
    batch_size,
    avg_seqlen,
    max_sequence_length,
    seqlen_stdev,
    vocab_size=30522,
    pad_idx=0,
):
    r"""
    Utility function to generate a batch of random sequences, together with their
    attention mask and lengths.
    Copied from: https://github.com/HamidShojanazeri/transformers/blob/ddf0299a13e7c4f54459a0731abd80204a1078f5/examples/pytorch/benchmarking/benchmark_bettertransformer.py#L149
    """
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)
    lengths = torch.clamp(lengths, min=0, max=max_sequence_length)

    tokens = torch.full(
        (batch_size, max_sequence_length),
        pad_idx,
    )
    # lengths[0:2] = max_sequence_length-1
    for i in range(batch_size):
        tokens[i, : lengths[i]] = torch.randint(
            pad_idx + 1,
            vocab_size - 1,
            size=(lengths[i],),  # type: ignore
        )

    if batch_size == 1:
        mask = None
    else:
        mask = torch.full(
            (batch_size, max_sequence_length),
            0,
        )
        for i in range(batch_size):
            mask[i, : lengths[i]] = 1

    return tokens, lengths, mask


def benchmark(
    model,
    input_ids,
    masks,
    num_batches,
    is_decoder,
    max_token,
    pad_token_id,
    warmup_steps: int = 3,
):
    # Warmup
    if is_decoder:
        gen_config = GenerationConfig(
            max_new_tokens=max_token,
            min_new_tokens=max_token,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
        for _ in range(warmup_steps):
            model.generate(
                input_ids, attention_mask=masks, generation_config=gen_config
            )
            torch.cuda.synchronize()

    else:
        for _ in range(warmup_steps):
            model(input_ids, masks)
            torch.cuda.synchronize()

    print("Warmup done...")

    # benchmark
    if is_decoder:
        total_time, max_mem = benchmark_transformers_with_memory(
            model, num_batches, input_ids, masks, is_decoder, gen_config  # type: ignore
        )
    else:
        total_time, max_mem = benchmark_transformers_with_memory(
            model, num_batches, input_ids, masks, is_decoder
        )

    return total_time, max_mem


def get_tokenizer(model_name: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(
    model_name: str, dtype: torch.dtype, use_cuda: bool = True
) -> AutoModel | AutoModelForCausalLM | AutoModelForSeq2SeqLM:
    task = TasksManager.infer_task_from_model(model_name)

    if task == "text-generation":
        autoclass = AutoModelForCausalLM
    elif task == "text2text-generation":
        autoclass = AutoModelForSeq2SeqLM
    else:
        autoclass = AutoModel

    if use_cuda:
        with torch.device("cuda:0"):
            model = autoclass.from_pretrained(model_name, torch_dtype=dtype)
    else:
        model = autoclass.from_pretrained(model_name, torch_dtype=dtype)
        model = model.to(dtype)

    return model


def get_inputs(
    model: AutoModel | AutoModelForCausalLM | AutoModelForSeq2SeqLM,
    batch_size: int,
    seq_len: int,
    pad_perc: float,
    device: torch.device,
    seqlen_stdev: int = 10,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    max_seqlen = seq_len
    mean_seqlen = int((1 - pad_perc) * max_seqlen)
    input_ids, _, attention_mask = get_batch(
        batch_size,
        mean_seqlen,
        max_seqlen,
        seqlen_stdev,
        vocab_size=model.config.vocab_size,  # type: ignore
    )
    input_ids = input_ids.to(device)
    if torch.is_tensor(attention_mask):
        attention_mask = attention_mask.to(device)  # type: ignore

    return input_ids, attention_mask


def cleanup(
    model: AutoModel | AutoModelForCausalLM | AutoModelForSeq2SeqLM,
    input_ids: torch.Tensor,
    masks: torch.Tensor | None,
):
    model.to("cpu")
    input_ids.detach().to("cpu")
    if torch.is_tensor(masks):
        masks.detach().to("cpu")

    del model
    del input_ids
    del masks
    torch.cuda.empty_cache()
    garbage_collect()


def run_model_forward_benchmark(
    model_name: str,
    is_decoder: bool,
    num_batches: int,
    cpu_bench: bool = False,
    batch_sizes: list[int] = BATCH_SIZES,
    num_tokens_list: list[int] = NUM_TOKENS,
    dtypes: list[torch.dtype] = DTYPES,
    sdp_backends: list[SDPBackend] = SDP_BACKENDS,
    seq_lens: list[int] = SEQ_LEN,
) -> list[ModelForwardRecord]:
    records: list[ModelForwardRecord] = []
    try:
        tokenizer = get_tokenizer(model_name)
        model = get_model(model_name, torch.float16, False)
    except Exception as e:
        print(e)
        print(f"Pipeline not found for {model_name}...")
        return records

    pad_percentage = 0
    # if is_decoder:
    #     PAD_PERCENTAGES = [0]
    # else:
    #     PAD_PERCENTAGES = [0, 0.1, 0.2, 0.5, 0.75]

    if cpu_bench:
        with torch.inference_mode():
            cpu_max_token = 256
            cpu_batch_size = 4
            cpu_seq_len = 64
            cpu_pad_percentage = 0
            input_ids, masks = get_inputs(
                model,
                cpu_batch_size,
                cpu_seq_len,
                cpu_pad_percentage,
                torch.device("cpu"),
            )

            try:
                time, max_mem = benchmark(
                    model,
                    input_ids,
                    masks,
                    num_batches,
                    is_decoder,
                    cpu_max_token,
                    tokenizer.pad_token_id,
                )
            except RuntimeError as e:
                time, max_mem = -1.0, -1
                print(e)

            records.append(
                ModelForwardRecord(
                    model=model_name,
                    num_batches=num_batches,
                    batch_size=cpu_batch_size,
                    seq_len=cpu_seq_len,
                    pad_percentage=cpu_pad_percentage,
                    num_tokens=cpu_max_token,
                    time=time,
                    max_mem=max_mem,
                    sdp_backend="CPU",
                    dtype=str(model.dtype),  # type: ignore
                    bettertransformer=False,
                    gpu="CPU",
                    compile=False,
                )
            )

            del input_ids
            del masks

    del model

    device = torch.device("cuda:0")
    with torch.inference_mode():
        for params in tqdm(
            list(itertools.product(batch_sizes, dtypes, seq_lens, num_tokens_list))
        ):
            batch_size, dtype, seq_len, num_token = params
            model = get_model(model_name, dtype, True)
            input_ids, masks = get_inputs(
                model, batch_size, seq_len, pad_percentage, device
            )

            try:
                time, max_mem = benchmark(
                    model,
                    input_ids,
                    masks,
                    num_batches,
                    is_decoder,
                    num_token,
                    tokenizer.pad_token_id,
                )
            except RuntimeError as e:
                time, max_mem = -1.0, -1
                print(e)
            finally:
                cleanup(model, input_ids, masks)

            records.append(
                ModelForwardRecord(
                    model=model_name,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    pad_percentage=pad_percentage,
                    num_tokens=num_token,
                    time=time,
                    max_mem=max_mem,
                    sdp_backend="EAGER",
                    dtype=str(dtype),
                    bettertransformer=False,
                    gpu=torch.cuda.get_device_name(),
                    compile=False,
                )
            )

    with torch.inference_mode():
        for params in tqdm(
            list(
                itertools.product(
                    sdp_backends,
                    dtypes,
                    BETTERTRANSFORMER,
                    COMPILE,
                    batch_sizes,
                    num_tokens_list,
                    seq_lens,
                )
            )
        ):
            (
                sdp_backend,
                dtype,
                bettertransformer,
                compile,
                batch_size,
                num_token,
                seq_len,
            ) = params
            model = get_model(model_name, dtype, True)
            input_ids, masks = get_inputs(
                model, batch_size, seq_len, pad_percentage, device
            )

            with Optimizer(sdp_backend, dtype, bettertransformer, compile) as opt:
                model = opt(model)

                try:
                    time, max_mem = benchmark(
                        model,
                        input_ids,
                        masks,
                        num_batches,
                        is_decoder,
                        num_token,
                        tokenizer.pad_token_id,
                    )
                except RuntimeError as e:
                    time, max_mem = -1.0, -1
                    print(e)
                finally:
                    cleanup(model, input_ids, masks)

            records.append(
                ModelForwardRecord(
                    model=model_name,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    pad_percentage=pad_percentage,
                    num_tokens=num_token,
                    time=time,
                    max_mem=max_mem,
                    sdp_backend=opt.sdp_backend.name
                    if sdp_backend != SDPBackend.ERROR
                    else "NATIVE",
                    dtype=str(dtype),
                    bettertransformer=bettertransformer,
                    gpu=torch.cuda.get_device_name(),
                    compile=opt.compile,
                    compile_mode=opt.compile_mode if opt.compile else None,
                )
            )

    return records


def get_model_forward_benchmark_df(
    model_name: str,
    num_batches: int = 32,
    cpu_bench: bool = False,
    batch_sizes: list[int] = BATCH_SIZES,
    num_tokens_list: list[int] = NUM_TOKENS,
    dtypes: list[torch.dtype] = DTYPES,
    sdp_backends: list[SDPBackend] = SDP_BACKENDS,
    seq_lens: list[int] = SEQ_LEN,
) -> pd.DataFrame:
    records = run_model_forward_benchmark(
        model_name=model_name,
        is_decoder=True,
        num_batches=num_batches,
        cpu_bench=cpu_bench,
        batch_sizes=batch_sizes,
        num_tokens_list=num_tokens_list,
        dtypes=dtypes,
        sdp_backends=sdp_backends,
        seq_lens=seq_lens,
    )
    return pd.DataFrame([r.__dict__ for r in records])
