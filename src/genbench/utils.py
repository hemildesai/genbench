import gc
import operator
from functools import reduce

import torch
import torch.utils.benchmark as benchmark
from torch.backends.cuda import SDPBackend
from tqdm import tqdm

# Helpful arguments mapper
BACKEND_MAP = {
    SDPBackend.MATH: {
        "enable_math": True,
        "enable_flash": False,
        "enable_mem_efficient": False,
    },
    SDPBackend.FLASH_ATTENTION: {
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": False,
    },
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False,
        "enable_flash": False,
        "enable_mem_efficient": True,
    },
}


def benchmark_torch_function(f, label: str, sub_label: str, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        label=label,
        sub_label=sub_label,
    ).blocked_autorange()
    return t0


def benchmark_torch_function_in_milliseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e3


def benchmark_function(
    func, n_repeat: int, use_torch_benchmark: bool, **kwargs
) -> float:
    if use_torch_benchmark:
        return benchmark_torch_function_in_milliseconds(func, **kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()  # type: ignore
    for _ in range(n_repeat):
        func(**kwargs)
    end_event.record()  # type: ignore

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time = (start_event.elapsed_time(end_event)) / n_repeat
    return time


def benchmark_transformers_with_memory(
    model, num_batches, input_ids, masks, is_decoder, generation_config=None
) -> tuple[float, int]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()  # type: ignore
    for _ in tqdm(range(num_batches)):
        if is_decoder:
            model.generate(
                input_ids, attention_mask=masks, generation_config=generation_config
            )
        else:
            model(input_ids, masks)
    end_event.record()  # type: ignore

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    max_memory = torch.cuda.max_memory_allocated()

    return start_event.elapsed_time(end_event) / num_batches, max_memory


def garbage_collect() -> None:
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                del obj.data
                del obj
        except:
            pass

    torch.cuda.empty_cache()


def get_gc_tensors() -> list[torch.Tensor]:
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                print(
                    reduce(operator.mul, obj.size()) if len(obj.size()) > 0 else 0,
                    type(obj),
                    obj.size(),
                )
                tensors.append(obj)
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                print(
                    reduce(operator.mul, obj.data.size())
                    if len(obj.data.size()) > 0
                    else 0,
                    type(obj),
                    obj.data.size(),
                )
                tensors.append(obj.data)
        except:
            pass

    return tensors
