import itertools
from typing import cast

import pandas as pd
import torch
from torch.backends.cuda import SDPBackend
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline, set_seed
from transformers.pipelines.base import no_collate_fn, pad_collate_fn
from transformers.pipelines.pt_utils import PipelineIterator
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from genbench.llm.fschat_conversation import conv_templates, get_conv_template
from genbench.llm.utils import TextGenerationPipelineRecord, get_questions
from genbench.optimizer import Optimizer
from genbench.utils import benchmark_function, garbage_collect

set_seed(42)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore

SDP_BACKENDS = [
    SDPBackend.ERROR,  # This means no kernel specific context manager
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
]
DTYPES = [torch.float16, torch.bfloat16]
BETTERTRANSFORMER = [True]
COMPILE = [True, False]
BATCH_SIZES = [1, 4, 8]
NUM_TOKENS = [256, 512]


def _get_prompts(batch_size: int, model_name: str) -> list[str]:
    questions = get_questions()[:batch_size]
    prompts = []
    for q in questions:
        conv = conv_templates.get(model_name, get_conv_template("one_shot"))
        conv.append_message(conv.roles[0], q.text)
        conv.append_message(conv.roles[1], None)  # type: ignore
        prompt = conv.get_prompt()
        prompts.append(prompt)

    return prompts


def run_warmup(
    pipeline: TextGenerationPipeline | Text2TextGenerationPipeline,
    warmup_steps: int = 3,
):
    prompts = _get_prompts(1, pipeline.model.config.model_type)
    for _ in range(warmup_steps):
        pipeline(prompts, max_length=1024, num_return_sequences=1)

    print("Warmup done...")


def get_pipeline(
    model_name: str, batch_size: int, device: int | str, dtype: torch.dtype | None
) -> TextGenerationPipeline | Text2TextGenerationPipeline:
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=device,
        batch_size=batch_size,
        torch_dtype=dtype,
    )

    generator = cast(TextGenerationPipeline | Text2TextGenerationPipeline, generator)

    if batch_size > 1:
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id  # type: ignore
        generator.tokenizer.padding_side = "left"  # type: ignore

    return generator


def run_generator(
    generator: Text2TextGenerationPipeline | TextGenerationPipeline,
    prompts: list[str],
    num_tokens: int,
    forward_only: bool,
    batch_size: int,
    n_repeat: int,
) -> float:
    if forward_only:
        kwargs = {
            "max_new_tokens": num_tokens,
            "min_new_tokens": num_tokens,
            "num_return_sequences": 1,
        }
        (
            preprocess_params,
            forward_params,
            _,
        ) = generator._sanitize_parameters(**kwargs)

        if len(prompts) == 1:
            model_inputs = generator.preprocess(prompts[0], **preprocess_params)
        else:
            dataset = PipelineIterator(prompts, generator.preprocess, preprocess_params)
            feature_extractor = (
                generator.feature_extractor
                if generator.feature_extractor is not None
                else generator.image_processor
            )
            collate_fn = (
                no_collate_fn
                if batch_size == 1
                else pad_collate_fn(generator.tokenizer, feature_extractor)
            )
            dataloader = DataLoader(
                dataset, num_workers=1, batch_size=batch_size, collate_fn=collate_fn
            )
            # Get next batch from Dataloader
            model_inputs = next(iter(dataloader))

        time = benchmark_function(
            generator.forward,
            n_repeat=n_repeat,
            use_torch_benchmark=False,
            model_inputs=model_inputs,
            **forward_params,
        )

    else:
        time = benchmark_function(
            generator,
            n_repeat=n_repeat,
            use_torch_benchmark=False,
            text_inputs=prompts,
            max_length=1024,
            num_return_sequences=1,
        )
    return time


def cleanup(
    pipeline: TextGenerationPipeline | Text2TextGenerationPipeline,
):
    pipeline.model.to("cpu")

    del pipeline.model
    del pipeline
    torch.cuda.empty_cache()
    garbage_collect()


def run_text_generation_benchmark(
    model_name: str = "gpt2",
    n_repeat: int = 10,
    forward_only: bool = False,
    cpu_bench: bool = False,
    batch_sizes: list[int] = BATCH_SIZES,
    num_tokens_list: list[int] = NUM_TOKENS,
    dtypes: list[torch.dtype] = DTYPES,
    sdp_backends: list[SDPBackend] = SDP_BACKENDS,
) -> list[TextGenerationPipelineRecord]:
    records: list[TextGenerationPipelineRecord] = []
    try:
        generator = get_pipeline(model_name, 1, "cpu", None)
    except Exception as e:
        print(e)
        print(f"Pipeline not found for {model_name}...")
        return records

    if cpu_bench:
        prompts = _get_prompts(1, model_name)
        try:
            with torch.inference_mode():
                run_warmup(generator)
                time = run_generator(generator, prompts, 256, forward_only, 1, 4)
        except RuntimeError as e:
            time = -1.0
            print(e)

        llm_record = TextGenerationPipelineRecord(
            batch_size=1,
            model=model_name,
            sdp_backend="CPU",
            dtype=str(generator.model.dtype),
            bettertransformer=False,
            time=time,
            gpu="CPU",
            prompts=prompts,
            forward_only=forward_only,
            compile=False,
            num_tokens=256,
        )
        records.append(llm_record)

    del generator.model
    del generator

    # EAGER mode
    with torch.inference_mode():
        for params in tqdm(
            list(itertools.product(dtypes, batch_sizes, num_tokens_list))
        ):
            dtype, batch_size, num_tokens = params
            if batch_size > len(get_questions()):
                continue

            prompts = _get_prompts(batch_size, model_name)
            generator = get_pipeline(model_name, batch_size, "cuda:0", dtype)

            try:
                run_warmup(generator)
                time = run_generator(
                    generator, prompts, num_tokens, forward_only, batch_size, n_repeat
                )
            except RuntimeError as e:
                time = -1.0
                print(e)
            finally:
                cleanup(generator)

            llm_record = TextGenerationPipelineRecord(
                batch_size=batch_size,
                num_tokens=num_tokens,
                model=model_name,
                sdp_backend="EAGER",
                dtype=str(dtype),
                bettertransformer=False,
                time=time,
                gpu=torch.cuda.get_device_name(),
                prompts=prompts,
                forward_only=forward_only,
                compile=False,
            )
            records.append(llm_record)

    # SDP mode
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
                )
            )
        ):
            (
                sdp_backend,
                dtype,
                bettertransformer,
                compile,
                batch_size,
                num_tokens,
            ) = params
            if batch_size > len(get_questions()):
                continue

            prompts = _get_prompts(batch_size, model_name)
            generator = get_pipeline(model_name, batch_size, "cuda:0", dtype)

            with Optimizer(sdp_backend, dtype, bettertransformer, compile) as opt:
                generator.model = opt(generator.model)

                try:
                    run_warmup(generator)
                    time = run_generator(
                        generator,
                        prompts,
                        num_tokens,
                        forward_only,
                        batch_size,
                        n_repeat,
                    )
                except RuntimeError as e:
                    print(e)
                    time = -1.0
                finally:
                    cleanup(generator)

            llm_record = TextGenerationPipelineRecord(
                batch_size=batch_size,
                num_tokens=num_tokens,
                model=model_name,
                sdp_backend=opt.sdp_backend.name
                if sdp_backend != SDPBackend.ERROR
                else "NATIVE",
                dtype=str(opt.dtype),
                bettertransformer=opt.bettertransformer,
                time=time,
                gpu=torch.cuda.get_device_name(),
                prompts=prompts,
                forward_only=forward_only,
                compile=opt.compile,
                compile_mode=opt.compile_mode if opt.compile else None,
            )
            records.append(llm_record)

    return records


def get_text_generation_benchmark_df(
    model_name: str,
    n_repeat: int = 8,
    forward_only: bool = False,
    cpu_bench: bool = False,
    batch_sizes: list[int] = BATCH_SIZES,
    num_tokens_list: list[int] = NUM_TOKENS,
    dtypes: list[torch.dtype] = DTYPES,
    sdp_backends: list[SDPBackend] = SDP_BACKENDS,
) -> pd.DataFrame:
    records = run_text_generation_benchmark(
        model_name=model_name,
        n_repeat=n_repeat,
        forward_only=forward_only,
        cpu_bench=cpu_bench,
        batch_sizes=batch_sizes,
        num_tokens_list=num_tokens_list,
        dtypes=dtypes,
        sdp_backends=sdp_backends,
    )
    return pd.DataFrame([r.__dict__ for r in records])


def profile_text_generation(
    model_name: str,
    n_repeat: int = 1,
    device: str = "cuda:0",
    sdp_backend: SDPBackend = SDPBackend.FLASH_ATTENTION,
    dtype: torch.dtype = torch.float16,
) -> profile | None:
    prompts = _get_prompts(1, model_name)
    try:
        generator = pipeline("text-generation", model=model_name, device=device)

        with Optimizer(sdp_backend, dtype, True, True) as opt, torch.inference_mode():
            generator.model = opt(generator.model)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for i in range(n_repeat):
                    with record_function(f"model_inference_{i}"):
                        generator(prompts, max_length=1024, num_return_sequences=1)
    except RuntimeError as e:
        print(e)
        return None

    return prof


def profile_text_generation_w_tensorboard(
    model_name: str,
    n_repeat: int = 3,
    device: str = "cuda:0",
    sdp_backend: SDPBackend = SDPBackend.FLASH_ATTENTION,
    dtype: torch.dtype = torch.float16,
) -> profile | None:
    assert n_repeat >= 3, "Need at least 3 iterations for profiling"
    prompts = _get_prompts(1, model_name)
    generator = pipeline("text-generation", model=model_name, device=device)

    try:
        with Optimizer(sdp_backend, dtype, True, True) as opt, torch.inference_mode():
            generator.model = opt(generator.model)

            run_warmup(
                cast(TextGenerationPipeline | Text2TextGenerationPipeline, generator)
            )
            with profile(
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=n_repeat - 2, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./log/{model_name}"
                ),
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for i in range(n_repeat):
                    with record_function(f"model_inference_{i}"):
                        generator(prompts, max_length=1024, num_return_sequences=1)

                    prof.step()
    except RuntimeError as e:
        print(e)
        return None

    return prof
