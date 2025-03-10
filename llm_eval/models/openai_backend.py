import openai
import time
import base64
import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger

logger = get_logger(name="openai_backend", level=logging.INFO)


@register_model("openai")
class OpenAIModel(BaseModel):
    """
    A production-grade OpenAI API backend model that supports multimodal inputs,
    chain-of-thought (CoT) prompting, and concurrent batch processing using multi-threading.

    This implementation has been updated to use the OpenAI SDK client, which provides a 
    more explicit client instance for API calls. It is compatible with both the official 
    OpenAI API and OpenAI-compatible servers (e.g., vLLM) by making the API key optional.

    Key Features:
      - Constructs payloads for both Chat and Completions API calls.
      - Processes image inputs by converting URLs or base64-encoded images to the required format.
      - Implements robust retry logic with exponential backoff.
      - Executes API calls concurrently using a ThreadPoolExecutor; the number of worker threads 
        is set based on the batch_size.
      - Supports chain-of-thought (CoT) prompting and parsing.

    Args:
        api_key (Optional[str]): OpenAI API key (optional if using an OpenAI-compatible server).
        api_base (str): Base URL for the API.
        model_name (str): Model identifier (e.g., "gpt-4", "gpt-4-vision-preview").
        system_message (Optional[str]): System message for chat completions.
        use_chat_api (bool): Whether to use the Chat API; if False, uses the Completions API.
        is_vision_model (bool): Flag indicating if the model supports vision inputs.
        cot_trigger (Optional[str]): A trigger phrase to induce chain-of-thought; if provided, appended to the prompt.
        cot_parser (Optional[Callable[[str], Tuple[str, str]]]): Function that parses generated text into (chain_of_thought, final_answer).
        batch_size (int): Number of concurrent API calls (worker threads) for batch processing.
        max_retries (int): Maximum number of retry attempts for API calls.
        cot (bool): Flag to enable chain-of-thought prompting.
        **kwargs: Additional API parameters (e.g., temperature, top_p, max_tokens, etc.).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = None,
        model_name: str = None,
        system_message: Optional[str] = None,
        use_chat_api: bool = True,
        is_vision_model: bool = False,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = None,
        batch_size: int = 8,
        max_retries: int = 3,
        cot: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_name or not api_base:
            raise ValueError("model_name and api_base are required")
        
        # Set up the OpenAI API client using the latest SDK. If api_key is None, this allows for usage
        # with OpenAI-compatible servers such as vLLM.
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        
        self.model_name = model_name
        self.system_message = system_message
        self.use_chat_api = use_chat_api
        self.is_vision_model = is_vision_model

        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cot = cot
        
        # Default API parameters (e.g., temperature, max_tokens, top_p, etc.)
        self.default_params = kwargs

    def _process_image_content(self, content: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Processes image content into the format expected by the OpenAI Vision API.

        Supports URLs, base64 strings, or dictionaries with detailed specifications.
        """
        VALID_DETAILS = {"high", "low", "auto"}
        
        def validate_detail(detail: str) -> str:
            detail = detail.lower() if detail else "auto"
            return detail if detail in VALID_DETAILS else "auto"
        
        def process_base64(b64_str: str, mime_type: str = "image/jpeg") -> str:
            try:
                b64_bytes = base64.b64decode(b64_str)
                if len(b64_bytes) > 20 * 1024 * 1024:
                    raise ValueError("Image size exceeds 20MB limit")
                return f"data:{mime_type};base64,{b64_str}"
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {str(e)}")
        
        if isinstance(content, list):
            max_images = self.default_params.get("max_images", float("inf"))
            if len(content) > max_images:
                raise ValueError(f"Number of images exceeds limit ({max_images})")
            return [self._process_image_content(item) for item in content]
        
        if isinstance(content, str):
            if content.startswith(("http://", "https://")):
                return {"type": "image_url", "image_url": {"url": content, "detail": "auto"}}
            try:
                return {"type": "image_url", "image_url": {"url": process_base64(content), "detail": "auto"}}
            except:
                return {"type": "text", "text": content}
        
        elif isinstance(content, dict):
            detail = validate_detail(content.get("detail", "auto"))
            if "image_url" in content:
                if isinstance(content["image_url"], str):
                    return {"type": "image_url", "image_url": {"url": content["image_url"], "detail": detail}}
                return {"type": "image_url", "image_url": {**content["image_url"], "detail": detail}}
            elif "base64" in content:
                mime_type = content.get("mime_type", "image/jpeg")
                return {"type": "image_url", "image_url": {"url": process_base64(content["base64"], mime_type), "detail": detail}}
        return {"type": "text", "text": str(content)}
    
    def _create_payload(
        self,
        inputs: Union[str, List[Dict], Dict],
        cot: bool = False,
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs the payload for an API call.

        Supports both Chat and Completions API calls. If chain-of-thought (CoT) is enabled
        (via the 'cot' parameter), the CoT trigger is appended to the prompt.
        Additionally, if 'until' is provided, it is added as a stop sequence.
        """
        params = deepcopy(self.default_params)
        params.update(kwargs)

        payload = {}
        if self.use_chat_api:
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            if isinstance(inputs, str):
                prompt_text = inputs
                if cot and self.cot_trigger:
                    prompt_text += f"\n{self.cot_trigger}\n"
                messages.append({"role": "user", "content": prompt_text})
            elif isinstance(inputs, list):
                messages.extend(inputs)
            else:
                messages.append({"role": "user", "content": str(inputs)})
            payload = {"model": self.model_name, "messages": messages}
            if until is not None:
                # Ensure 'until' is a list of strings
                if isinstance(until, str):
                    until = [until]
                payload["stop"] = until
        else:
            prompt_text = inputs if not (cot and self.cot_trigger) else f"{inputs}\n{self.cot_trigger}\n"
            payload = {"model": self.model_name, "prompt": prompt_text}
            if params.get("logprobs") is not None:
                payload["logprobs"] = params["logprobs"]
            if until is not None:
                if isinstance(until, str):
                    until = [until]
                payload["stop"] = until

        # Set additional API parameters (e.g., max_tokens, temperature, top_p, etc.)
        for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in params:
                payload[param] = params[param]
        
        # Remove any keys with None values.
        return {k: v for k, v in payload.items() if v is not None}

    def _generate_single(
        self,
        item: Dict[str, Any],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool = False,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates text for a single input item with retry logic and exponential backoff.

        Args:
            item (Dict[str, Any]): An input item containing at least the "input" key.
            return_logits (bool): Whether to return logits.
            until (Optional[Union[str, List[str]]]): Optional stop criteria.
            cot (bool): Whether to enable chain-of-thought prompting.
            max_retries (int | None): Maximum retry attempts for this call.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            Dict[str, Any]: A dictionary with keys "prediction", "finish_reason", and optionally "logprobs".
        """
        effective_retries = max_retries if max_retries is not None else self.max_retries
        for attempt in range(effective_retries):
            try:
                payload = self._create_payload(item["input"], cot=cot, until=until, **kwargs)
                if not self.use_chat_api:
                    response = self.client.completions.create(**payload)
                    result = {
                        "prediction": response.choices[0].text,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits:
                        result.update({
                            "logprobs": response.choices[0].logprobs.token_logprobs,
                            "tokens": response.choices[0].logprobs.tokens,
                        })
                else:
                    response = self.client.chat.completions.create(**payload)
                    result = {
                        "prediction": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits and hasattr(response.choices[0], "logprobs"):
                        result["logprobs"] = response.choices[0].logprobs
                if cot and self.cot_parser:
                    generated_text = result["prediction"]
                    cot_text, final_answer = self.cot_parser(generated_text)
                    result["chain_of_thought"] = cot_text
                    result["prediction"] = final_answer
                return result
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{effective_retries} failed: {e}")
                time.sleep(min(2 ** attempt, 32))
        raise RuntimeError(f"Failed after {effective_retries} retries.")

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        until: Optional[Union[str, List[str]]] = None,
        cot: bool = False,
        max_retries: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of input items using multi-threading.

        This method processes each input item concurrently using a ThreadPoolExecutor.
        It supports chain-of-thought prompting and parsing. In case of API failures,
        each call is retried with exponential backoff.

        Args:
            inputs (List[Dict[str, Any]]): A list of input items. Each item must include at least the "input" key,
                                           and optionally a "reference" key.
            return_logits (bool): If True, includes logits in the output (if supported).
            until (Optional[Union[str, List[str]]]): Stop sequence(s) for generation.
            cot (bool): If True, enables chain-of-thought processing.
            max_retries (int | None): Maximum retry attempts for each API call (defaults to self.max_retries).
            show_progress (bool): If True, displays a progress bar.
            **kwargs: Additional API parameters.

        Returns:
            List[Dict[str, Any]]: A list of input items updated with generation results.
                                  Each item will have "prediction" and "finish_reason" keys,
                                  and if CoT is enabled, "chain_of_thought" will also be added.
        """
        if max_retries is None:
            max_retries = self.max_retries

        results = []
        max_workers = self.batch_size

        future_to_item = {}

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            input_copy = deepcopy(item)
            return self._generate_single(input_copy, return_logits, until, cot=cot, max_retries=max_retries, **kwargs)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for item in inputs:
                future = executor.submit(process_item, item)
                future_to_item[future] = deepcopy(item)
            for future in tqdm(as_completed(future_to_item), total=len(inputs), desc="Generating OpenAI outputs", disable=not show_progress):
                orig_item = future_to_item[future]
                try:
                    res = future.result()
                    merged = deepcopy(orig_item)
                    merged.update(res)
                    results.append(merged)
                except Exception as e:
                    logger.error(f"Error in API call: {str(e)}")
                    error_item = deepcopy(orig_item)
                    error_item.update({
                        "error": str(e),
                        "prediction": None,
                        "finish_reason": "error"
                    })
                    results.append(error_item)
        return results
