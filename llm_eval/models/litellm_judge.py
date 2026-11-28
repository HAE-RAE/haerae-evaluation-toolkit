import asyncio
import logging
from typing import List, Dict, Any, Optional, Union

import litellm
from tqdm.asyncio import tqdm_asyncio

from . import register_model
from .base import BaseJudge
from llm_eval.utils.logging import get_logger

logger = get_logger(name="litellm_judge", level=logging.INFO)


@register_model("litellm_judge")
class LiteLLMJudge(BaseJudge):
    """
    Async judge backend using LiteLLM acompletion with retries.

    Args:
        provider: LLM provider (e.g., "openai", "azure", "bedrock").
        model_name: Model/deployment name.
        api_key: Provider API key.
        api_base: Base URL.
        api_version: API version (Azure).
        max_new_tokens: Max tokens for judge response.
        temperature: Sampling temperature.
        batch_size: Concurrency for judge requests.
        retry_max: Max retries per request.
        retry_base_delay: Base backoff delay.
        **kwargs: Extra LiteLLM params.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        batch_size: int = 8,
        retry_max: int = 3,
        retry_base_delay: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.provider = (provider or "").lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.retry_max = max(0, retry_max)
        self.retry_base_delay = max(0.0, retry_base_delay)

        self.common_params: Dict[str, Any] = {}
        if api_key is not None:
            self.common_params["api_key"] = api_key
        if api_base is not None:
            self.common_params["api_base"] = api_base
        if api_version is not None:
            self.common_params["api_version"] = api_version

        self.extra_kwargs = kwargs or {}
        self.model_identifier = self._make_model_identifier(self.provider, self.model_name)
        logger.info(f"[LiteLLMJudge] Using provider='{self.provider}', model='{self.model_identifier}'")

    @staticmethod
    def _make_model_identifier(provider: str, model_name: str) -> str:
        if provider in {"azure", "bedrock"}:
            return f"{provider}/{model_name}"
        return model_name

    def _build_payload(self, prompt: str, until: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_identifier,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **self.common_params,
            **self.extra_kwargs,
        }
        if until is not None:
            payload["stop"] = until if isinstance(until, list) else [until]
        return {k: v for k, v in payload.items() if v is not None}

    async def _once(self, payload: Dict[str, Any]) -> str:
        resp = await litellm.acompletion(**payload)
        return resp.choices[0].message.content

    async def _with_retries(self, payload: Dict[str, Any]) -> str:
        attempt = 0
        last_e: Optional[Exception] = None
        while attempt <= self.retry_max:
            try:
                return await self._once(payload)
            except Exception as e:
                last_e = e
                if attempt == self.retry_max:
                    break
                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(f"[LiteLLMJudge] attempt={attempt+1} failed: {e}. retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                attempt += 1
        raise last_e or RuntimeError("LiteLLMJudge: exhausted retries")

    async def _judge_async(
        self,
        inputs: List[Dict[str, Any]],
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(self.batch_size if isinstance(self.batch_size, int) else 8)

        async def _worker(item: Dict[str, Any]) -> Dict[str, Any]:
            try:
                payload = self._build_payload(item.get("input", ""), until=until)
                async with sem:
                    prediction = await self._with_retries(payload)
                return {**item, "prediction": prediction}
            except Exception as exc:
                logger.error("[LiteLLMJudge] Judging failed: %s", exc)
                return {**item, "prediction": "", "error": str(exc), "finish_reason": "error"}

        return await tqdm_asyncio.gather(
            *(_worker(item) for item in inputs), disable=not show_progress,
            desc="LiteLLM Judge Batch",
        )

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return asyncio.run(self._judge_async(inputs, until=until, show_progress=show_progress))
