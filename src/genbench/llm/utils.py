import io
import urllib.request

import jsonlines
import torch
from pydantic import BaseModel

from genbench.llm.fschat_conversation import get_conv_template

QUESTION_PATHS = [
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/eval/table/question.jsonl"
]

CUDNN_VERSION = torch.backends.cudnn.version() if torch.cuda.is_available() else None  # type: ignore
CUDA_VERSION = torch.version.cuda if torch.cuda.is_available() else None  # type: ignore
PYTORCH_VERSION = torch.__version__  # type: ignore


# Create a pydantic model with fields question_id, text, category called Question
class Question(BaseModel):
    question_id: int
    text: str
    category: str


def get_questions() -> list[Question]:
    questions: list[Question] = []
    for path in QUESTION_PATHS:
        # Open file at the given url path
        with urllib.request.urlopen(path) as url:
            with io.TextIOWrapper(url, encoding="utf-8") as file:
                questions.extend(map(lambda o: Question(**o), jsonlines.Reader(file)))
    return questions


class TextGenerationPipelineRecord(BaseModel):
    model: str
    batch_size: int
    num_tokens: int
    time: float
    sdp_backend: str
    dtype: str
    bettertransformer: bool
    gpu: str
    forward_only: bool
    prompts: list[str]
    compile: bool
    compile_mode: str | None = None
    pytorch_version: str = PYTORCH_VERSION
    cuda_version: str | None = CUDA_VERSION
    cudnn_version: str | None = CUDNN_VERSION
    cloud: str = "LOCAL"


class TokenizerRecord(BaseModel):
    tokenizer: str
    num_threads: int
    num_bytes: int
    time: float
    throughput: float
    processor: str
    cores: int


class ModelForwardRecord(BaseModel):
    model: str
    num_batches: int
    batch_size: int
    seq_len: int
    pad_percentage: float
    num_tokens: int
    time: float
    max_mem: int
    sdp_backend: str
    dtype: str
    bettertransformer: bool
    gpu: str
    compile: bool
    compile_mode: str | None = None
    pytorch_version: str = PYTORCH_VERSION
    cuda_version: str | None = CUDA_VERSION
    cudnn_version: str | None = CUDNN_VERSION
    cloud: str = "LOCAL"
