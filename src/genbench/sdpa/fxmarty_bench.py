# This script is adopted from https://github.com/fxmarty/efficient-attention-benchmark/blob/main/benchmark.py
# It removes Hazy-Research implementation and adds PyTorch Native implementation without specifying a kernel.
# It also adds all three (math, flash, and efficient) implementations of PyTorch with specifying a kernel.
import itertools
import math
from typing import Dict, Iterable

import pandas as pd
import torch
import torch.backends.cuda
from tqdm import tqdm

from genbench.sdpa.utils import SDPARecord
from genbench.utils import benchmark_function


def attention_pytorch_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
):
    L = query.size(-1)
    S = value.size(-1)
    attn_mask = (
        torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    )
    if attn_mask is None:
        attn_weight = torch.softmax(
            (query @ key.transpose(-2, -1) / math.sqrt(L)), dim=-1
        )
    else:
        attn_weight = torch.softmax(
            (query @ key.transpose(-2, -1) / math.sqrt(L)) + attn_mask, dim=-1
        )
    attn_weight = torch.nn.functional.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def attention_pytorch_native(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
    )


def run_sdpa_fxmarty_benchmark() -> list[SDPARecord]:
    torch.manual_seed(0)
    n_repeat = 30
    dropout_p = 0.0
    causal = False
    dtype = (
        torch.float16
    )  # torch.float32 is not supported for Hazy-Research implementation
    device = "cuda"

    all_parameters = {
        "batch_size": [8, 16, 64],
        "seq_len": [64, 128, 256, 512, 1024],
        "head_dim": [32, 64, 128],
        "num_heads": [12, 16, 24],
    }

    def grid(
        parameters: Dict[str, list[int]],
    ) -> Iterable[list[int]]:
        for params in itertools.product(*parameters.values()):
            returned_list = list(params)
            yield returned_list

    output_file_path = "benchmark_attention.csv"
    output_file = open(output_file_path, "w")
    output_file.write(
        "batch_size, seq_len, headdim, nheads, PT eager (ms/forward), PT Native (ms/forward), PT Flash (ms/forward), PT Mem Eff (ms/forward), PT Math (ms/forward), Native speedup over eager, Flash speedup over eager, Mem Eff speedup over eager, Math speedup over eager\n"
    )
    records = []

    for params in tqdm(list(grid(all_parameters))):
        batch_size, seqlen, headdim, nheads = tuple(params)
        print(
            f"Running: bs={batch_size}, seqlen={seqlen}, headdim={headdim}, nheads={nheads}"
        )

        qkv = torch.randn(
            batch_size, 3, nheads, seqlen, headdim, device=device, dtype=dtype
        )
        query, key, value = qkv.unbind(dim=1)

        with torch.inference_mode():
            res_pt_eager = attention_pytorch_eager(query, key, value)
            res_pt_native = attention_pytorch_native(query, key, value)

            assert torch.allclose(
                res_pt_eager, res_pt_native, atol=5e-3
            ), f" Maxdiff: {(res_pt_eager - res_pt_native).abs().max()}"

            time_pt_eager = benchmark_function(
                attention_pytorch_eager,
                n_repeat=n_repeat,
                use_torch_benchmark=False,
                query=query,
                key=key,
                value=value,
                is_causal=False,
            )

            time_pt_native = benchmark_function(
                attention_pytorch_native,
                n_repeat=n_repeat,
                use_torch_benchmark=False,
                query=query,
                key=key,
                value=value,
                is_causal=False,
                # use_torch_benchmark=True,
            )

            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                time_flash = benchmark_function(
                    attention_pytorch_native,
                    n_repeat=n_repeat,
                    use_torch_benchmark=False,
                    query=query,
                    key=key,
                    value=value,
                    is_causal=False,
                )

            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                try:
                    time_mem_eff = benchmark_function(
                        attention_pytorch_native,
                        n_repeat=n_repeat,
                        use_torch_benchmark=False,
                        query=query,
                        key=key,
                        value=value,
                        is_causal=False,
                    )
                except RuntimeError as e:
                    print(f"Mem eff not applicable for {params}")
                    time_mem_eff = -1.0

            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                time_math = benchmark_function(
                    attention_pytorch_native,
                    n_repeat=n_repeat,
                    use_torch_benchmark=False,
                    query=query,
                    key=key,
                    value=value,
                    is_causal=False,
                )

            output_file.write(
                f"{batch_size},{seqlen},{headdim},{nheads},{time_pt_eager:.3f},{time_pt_native:.3f},{time_flash:.3f},{time_mem_eff:.3f},{time_math:.3f},{time_pt_eager / time_pt_native:.3f},{time_pt_eager / time_flash:.3f},{time_pt_eager / time_mem_eff:.3f},{time_pt_eager / time_math:.3f}\n"
            )
            record = SDPARecord(
                batch_size=batch_size,
                seqlen=seqlen,
                headdim=headdim,
                nheads=nheads,
                time_pt_eager=time_pt_eager,
                time_pt_native=time_pt_native,
                time_flash=time_flash,
                time_mem_eff=time_mem_eff,
                time_math=time_math,
            )
            records.append(record)

    output_file.close()
    return records


def get_sdpa__fxmarty_benchmark_df() -> pd.DataFrame:
    records = run_sdpa_fxmarty_benchmark()
    return pd.DataFrame([r.__dict__ for r in records])


if __name__ == "__main__":
    run_sdpa_fxmarty_benchmark()
