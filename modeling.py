from __future__ import annotations

import base64
import io
import os
import random
import time
from typing import Any, Sequence


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


PROVIDERS = ("openai", "gemini", "qwen35")


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def convert_image_to_text(image: Any) -> str:
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


class OpenAIModel:
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 1.0,
        timeout: int = 60,
        max_retries: int = 5,
        retry_sleep_seconds: float = 30.0,
        retry_backoff: float = 2.0,
        max_completion_tokens: int | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.retry_backoff = retry_backoff
        self.max_completion_tokens = max_completion_tokens
        self.client = None

    def load(self) -> None:
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use provider='openai'.") from exc

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=self.timeout,
        )

    def make_messages(self, prompt: str, image: Any = None) -> list[dict]:
        inputs = [{"type": "text", "text": prompt}]
        if image is not None:
            image_text = convert_image_to_text(image)
            url = f"data:image/png;base64,{image_text}"
            inputs.append({"type": "image_url", "image_url": {"url": url}})
        return [{"role": "user", "content": inputs}]

    def run(self, prompt: str, image: Any = None) -> str:
        self.load()
        error_message = "The response was filtered"

        for attempt in range(1, self.max_retries + 1):
            try:
                request_kwargs = {
                    "model": self.model_name_or_path,
                    "messages": self.make_messages(prompt, image),
                    "temperature": self.temperature,
                }
                if self.max_completion_tokens is not None:
                    request_kwargs["max_completion_tokens"] = self.max_completion_tokens

                response = self.client.chat.completions.create(**request_kwargs)
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content
                if output:
                    return output
                raise RuntimeError("OpenAI response was empty.")
            except Exception as exc:
                if error_message in str(exc):
                    return error_message
                if attempt >= self.max_retries:
                    print(f"OpenAIModel request failed after {attempt} attempts: {exc}")
                    return ""

                sleep_for = self.retry_sleep_seconds * (self.retry_backoff ** (attempt - 1))
                sleep_for += random.uniform(0, min(3.0, self.retry_sleep_seconds))
                print(
                    f"OpenAIModel request failed on attempt {attempt}/{self.max_retries}: {exc}. "
                    f"Sleeping {sleep_for:.1f}s before retrying."
                )
                time.sleep(sleep_for)

        return ""


class GeminiModel:
    def __init__(
        self,
        model_name_or_path: str = "gemini-3.1-flash-lite",
        temperature: float = 1.0,
        api_key: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.api_key = api_key
        self.client = None

    def load(self) -> None:
        if self.client is not None:
            return
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install the google-genai package to use provider='gemini'.") from exc

        key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is missing. Set it in .env or the environment.")
        self.client = genai.Client(api_key=key)

    def make_contents(self, prompt: str, image: Any = None) -> list:
        contents = []
        if image is not None:
            contents.append(image)
        contents.append(prompt)
        return contents

    def run(self, prompt: str, image: Any = None) -> str:
        self.load()

        try:
            response = self.client.models.generate_content(
                model=self.model_name_or_path,
                contents=self.make_contents(prompt, image),
                config={"temperature": self.temperature},
            )
            return response.text or ""
        except Exception as exc:
            print(f"Gemini API request failed: {exc}")
            return ""


class Qwen35Model:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        dtype: str | None = None,
        enable_thinking: bool | None = None,
        stop_on_answer: bool | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.tensor_parallel_size = (
            tensor_parallel_size
            if tensor_parallel_size is not None
            else _env_int("QWEN35_VLLM_TENSOR_PARALLEL_SIZE", 1)
        )
        self.gpu_memory_utilization = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else _env_float("QWEN35_VLLM_GPU_MEMORY_UTILIZATION")
        )
        self.max_model_len = (
            max_model_len
            if max_model_len is not None
            else _env_int("QWEN35_VLLM_MAX_MODEL_LEN")
        )
        self.max_num_seqs = (
            max_num_seqs
            if max_num_seqs is not None
            else _env_int("QWEN35_VLLM_MAX_NUM_SEQS")
        )
        self.dtype = dtype or os.getenv("QWEN35_VLLM_DTYPE", "auto")
        self.enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else _env_bool("QWEN35_VLLM_ENABLE_THINKING", False)
        )
        self.stop_on_answer = (
            stop_on_answer
            if stop_on_answer is not None
            else _env_bool("QWEN35_VLLM_STOP_ON_ANSWER", True)
        )
        self.llm = None
        self.processor = None
        self.sampling_params = None
        self.supports_multimodal = True

    def load(self, require_multimodal: bool = False) -> None:
        if self.llm is not None and self.processor is not None:
            return
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "Install vllm and transformers to use provider='qwen35'."
            ) from exc

        is_local = os.path.isdir(self.model_name_or_path)
        print(f"Loading Qwen3.5 vLLM model from {self.model_name_or_path}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        engine_kwargs: dict[str, Any] = {
            "model": self.model_name_or_path,
            "trust_remote_code": True,
            "dtype": self.dtype,
            "limit_mm_per_prompt": {"image": 1},
        }
        if self.tensor_parallel_size is not None:
            engine_kwargs["tensor_parallel_size"] = self.tensor_parallel_size
        if self.gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len
        if self.max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = self.max_num_seqs

        try:
            self.llm = LLM(**engine_kwargs)
        except (TypeError, ValueError) as exc:
            if "limit_mm_per_prompt" not in str(exc):
                raise
            engine_kwargs.pop("limit_mm_per_prompt", None)
            self.supports_multimodal = not (
                isinstance(exc, ValueError)
                and "only supported for multimodal models" in str(exc)
            )
            if not self.supports_multimodal:
                if require_multimodal:
                    raise RuntimeError(
                        "vLLM did not recognize this model as multimodal, but image "
                        "inputs were provided. Use a vLLM-supported VL checkpoint "
                        "or run this model without images."
                    ) from exc
                print(
                    "vLLM did not recognize this model as multimodal; "
                    "retrying without limit_mm_per_prompt."
                )
            self.llm = LLM(**engine_kwargs)

        sampling_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }
        if self.top_k is not None:
            sampling_kwargs["top_k"] = self.top_k
        if self.stop_on_answer:
            sampling_kwargs["stop"] = ["</answer>"]
            sampling_kwargs["include_stop_str_in_output"] = True
        try:
            self.sampling_params = SamplingParams(**sampling_kwargs)
        except TypeError as exc:
            if "include_stop_str_in_output" not in str(exc):
                raise
            sampling_kwargs.pop("include_stop_str_in_output", None)
            self.sampling_params = SamplingParams(**sampling_kwargs)

    def make_messages(self, prompt: str, image: Any = None) -> list[dict]:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if not self.enable_thinking and not prompt.lstrip().startswith("/no_think"):
            prompt = f"/no_think\n{prompt}"
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def make_vllm_input(self, prompt: str, image: Any = None) -> dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Qwen35Model.load() must be called before make_vllm_input().")
        if image is not None and not self.supports_multimodal:
            raise RuntimeError(
                "vLLM initialized this model as text-only, so image inputs cannot be sent. "
                "Use a VL model that your vLLM version supports, upgrade vLLM, or run this "
                "model without images."
            )

        messages = self.make_messages(prompt, image)
        try:
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        request: dict[str, Any] = {"prompt": prompt_text}
        if image is not None:
            request["multi_modal_data"] = {"image": image}
        return request

    def run(
        self,
        prompts: Sequence[str],
        images: Sequence[Any | None] | None = None,
    ) -> list[str]:
        prompts = list(prompts)
        if images is None:
            images = [None] * len(prompts)
        images = list(images)
        if len(prompts) != len(images):
            raise ValueError("prompts and images must have the same length.")
        if not prompts:
            return []
        self.load(require_multimodal=any(image is not None for image in images))

        requests = [
            self.make_vllm_input(prompt, image)
            for prompt, image in zip(prompts, images)
        ]
        outputs = self.llm.generate(requests, sampling_params=self.sampling_params)

        results = []
        for output in outputs:
            candidates = getattr(output, "outputs", None)
            if not candidates:
                results.append("")
                continue
            results.append(candidates[0].text or "")
        return results


def get_model(
    provider: str,
    model_name_or_path: str,
    timeout: int | None = None,
    max_retries: int | None = None,
    retry_sleep_seconds: float | None = None,
    retry_backoff: float | None = None,
    max_completion_tokens: int | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    dtype: str | None = None,
    enable_thinking: bool | None = None,
    stop_on_answer: bool | None = None,
) -> OpenAIModel | GeminiModel | Qwen35Model:
    """Build a model for one of PROVIDERS. Options a provider does not use are ignored."""
    if provider == "openai":
        kwargs = {
            "temperature": temperature,
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_sleep_seconds": retry_sleep_seconds,
            "retry_backoff": retry_backoff,
            "max_completion_tokens": max_completion_tokens,
        }
        return OpenAIModel(model_name_or_path, **{k: v for k, v in kwargs.items() if v is not None})
    if provider == "gemini":
        kwargs = {"temperature": temperature}
        return GeminiModel(model_name_or_path, **{k: v for k, v in kwargs.items() if v is not None})
    if provider == "qwen35":
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "dtype": dtype,
            "enable_thinking": enable_thinking,
            "stop_on_answer": stop_on_answer,
        }
        return Qwen35Model(model_name_or_path, **{k: v for k, v in kwargs.items() if v is not None})
    raise ValueError(f"Unknown provider: {provider}. Use one of: {', '.join(PROVIDERS)}.")
