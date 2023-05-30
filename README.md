# genbench

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/genbench.svg)](https://pypi.org/project/genbench) -->
<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/genbench.svg)](https://pypi.org/project/genbench) -->

---

An easy to use toolkit to run benchmarks on PyTorch based Generative models. Currently supports benchmarks for the following:

- Torch Scaled Dot Product Attention (SDPA) Kernel
- LLM Tokenizers (Tiktoken and Huggingface Tokenizer)
- Huggingface Transformers Text Generation Pipeline
- Huggingface Transformers Model forward call

It aims to collect benchmarks from a variety of sources (in addition to custom ones), including:

- [fxmarty/efficient-attention-benchmark](https://github.com/fxmarty/efficient-attention-benchmark)
- [optimum/benchmark_bettertransformer](https://github.com/huggingface/optimum/blob/main/tests/benchmark/benchmark_bettertransformer.py)
- [tiktoken/scripts/benchmark.py](https://github.com/openai/tiktoken/blob/main/scripts/benchmark.py)

The benchmarking functions return a dataframe with all necessary columns that can serve as a basis for further analysis.
The columns are consistent so different benchmarks can be combined into one dataframe. This can help to run analysis across a variety of different factors like GPUs, Models, Batch Sizes, Optimizations, CUDA and CUDNN versions, torch.compile use, etc. It can also serve as a way to quickly get the best optimizations for a model. It also provides utility functions to easily profile models and functions via the Torch Profiler.

Additionally, it also provides an easy to use Optimizer that allows you to apply a variety of optimizations to a model and benchmark it.

We provide CSV files for prerun benchmarks in the `benchmarks` folder. These can be used to quickly compare your results with ours. The folder will be updated regularly with new benchmarks. Example notebooks in the `notebooks` folder show how to quickly analyze the results.

The following optimizations are currently supported (More coming soon, including CUDA graphs, Torch Dynamo export, etc):

- Torch SDPA via Optimum BetterTransformer. `genbench` can run isolated benchmark for each SDPA kernel including Flash Attention, Efficient Attention, Math and Native (without kernel selection).
- Torch compile for Torch versions > 2.

The following precisions are currently supported (More coming soon, including 8bit and maybe 4bit?):

- torch.float32
- torch.float16
- torch.bfloat16

Please open an issue if you want to see your favorite optimization, precision or model supported.

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install git+https://github.com/hemildesai/genbench.git
```

If this doesn't work, you can clone the repo and do a manual install.

## Usage

_NOTE: More detailed documentation coming soon_

For now, the package does assume that you are running it on a GPU based system. It will be updated to show a warning if you are running it on a CPU based system.

Get benchmark df for Text Generation Pipeline:

```python
import genbench.llm.text_generation as textgen_bench
df = textgen_bench.get_text_generation_benchmark_df("gpt2", forward_only=True, cpu_bench=False, n_repeat=8)
```

Get profiler for Text Generation Pipeline:

```python
import genbench.llm.text_generation as textgen_bench
profiler = textgen_bench.profile_text_generation("gpt2")
```

Optimize a torch model:

```python
from genbench.optimizer import Optimizer
from torch.backends.cuda import SDPBackend
from transformers import AutoModel

model = AutoModel.from_pretrained("gpt2")
with Optimizer(sdp_backend=SDPBackend.FLASH_ATTENTION, dtype=torch.float16, bettertransformer=True, compile=True) as opt:
    model = opt(model)
    model(...)
```

See [llm_bench.ipynb](notebooks/llm_bench.ipynb) for a short notebook on how to analyze the dataframe.

## License

`genbench` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
