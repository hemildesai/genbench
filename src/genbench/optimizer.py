from types import TracebackType

import torch
from torch.backends.cuda import SDPBackend, sdp_kernel
from transformers import PreTrainedModel

from genbench.utils import BACKEND_MAP


class Optimizer:
    def __init__(
        self,
        sdp_backend: SDPBackend,
        dtype: torch.dtype = torch.float32,
        bettertransformer: bool = False,
        compile: bool = False,
    ):
        self.sdp_backend = sdp_backend
        if sdp_backend == SDPBackend.ERROR:
            self.sdp = None
        else:
            self.sdp = sdp_kernel(**BACKEND_MAP[sdp_backend])

        self.dtype = dtype
        self.bettertransformer = bettertransformer
        self.compile = compile
        self.compile_mode = "max-autotune"

    def __enter__(self):
        if self.sdp:
            self.sdp.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        if self.sdp:
            self.sdp.__exit__(exc_type, exc_val, exc_tb)
        if exc_type:
            print(exc_type, exc_val, exc_tb, sep="\n")

    def __call__(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.bettertransformer and hasattr(model, "to_bettertransformer"):
            model = model.to_bettertransformer()
        elif not hasattr(model, "to_bettertransformer"):
            self.bettertransformer = False
            print("Model does not support BetterTransformer")

        model = model.to(dtype=self.dtype)

        if self.compile:
            try:
                model = torch.compile(model, mode=self.compile_mode)  # type: ignore
            except RuntimeError as e:
                print(f"Compilation failed: {e}")
                self.compile = False

        return model
