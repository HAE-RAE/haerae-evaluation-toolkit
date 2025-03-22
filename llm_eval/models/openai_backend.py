import openai
import asyncio
import time
import base64
import logging
import json
import random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import httpx
from tqdm import tqdm

from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger

logger = get_logger(name="openai_backend", level=logging.INFO)


@register_model("openai")
class OpenAIModel(BaseModel):
    """
    A production-grade OpenAI API backend model that supports multimodal input,
    chain-of-thought (CoT) prompting, and concurrent batch processing using multi-threading.
    
    This implementation handles both Chat and Completions API calls based on the `use_chat_api` flag.
    It appends a CoT trigger if enabled and uses a CoT parser to extract a chain-of-thought and final answer.
    
    Key Features:
      - Constructs payloads for both Chat and traditional Completion API calls.
      - Processes image inputs by converting URLs or base64-encoded images to the required format.
      - Implements robust retry logic with exponential backoff.
      - Executes API calls concurrently using a ThreadPoolExecutor; the number of worker threads is set based on the batch_size.
      - Rich error handling and detailed logging for production monitoring.
    
    Args:
        api_key (str): OpenAI API key.
        api_base (str): OpenAI API base URL.
        model_name (str): Model identifier (e.g., "gpt-4", "gpt-4-vision-preview").
        model_name_or_path (str): Model identifier or path (e.g., "gpt-4", "gpt-4-vision-preview").
        system_message (Optional[str]): System message for chat completions.
        use_chat_api (bool): Whether to use the Chat API; if False, uses the Completions API.
        is_vision_model (bool): Flag indicating if the model supports vision inputs.
        cot_trigger (Optional[str]): A trigger phrase to induce chain-of-thought; if provided, appended to the prompt.
        cot_parser (Optional[Callable[[str], Tuple[str, str]]]): Function that parses generated text into (chain_of_thought, final_answer).
        batch_size (int): Number of concurrent API calls (worker threads) for batch processing.
        max_retries (int): Maximum number of retry attempts for API calls.
        **kwargs: Additional API parameters (e.g., temperature, top_p, max_tokens, etc.).
    """
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        stop: Optional[Union[str, List[str]]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        retries: int = 3,
        timeout: int = 60,
        delay: float = 0.5,
        **kwargs
    ):
        """
        Initialize an OpenAI model backend with specified parameters.
        """
        super().__init__(**kwargs)
        
        # 모델 이름 설정
        self.model_name = f"openai:{model_name}"
        
        # Set up OpenAI API credentials
        if api_key:
            openai.api_key = api_key
        if api_base:
            openai.api_base = api_base
        
        # Default API parameters (e.g., temperature, max_tokens, top_p, etc.)
        self.default_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
        }

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
        return_logits: bool = False,
        use_chat_api: Optional[bool] = None,
        until: Optional[Union[str, List[str]]] = None,
        cot: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs the payload for an API call.
        
        Supports both Chat and Completions APIs. If chain-of-thought (CoT) is enabled
        (determined by the explicit parameter 'cot'), the CoT trigger is appended to the prompt.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        use_chat_api = self.use_chat_api if use_chat_api is None else use_chat_api
        
        payload = {"model": self.model_name}
        
        # Add stop sequences if provided
        if until is not None:
            if isinstance(until, str):
                until = [until]
            payload["stop"] = until
        
        if use_chat_api:
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            # Process input: append CoT trigger if CoT is enabled via the explicit 'cot' parameter
            if isinstance(inputs, str):
                prompt_text = inputs
                if cot and self.cot_trigger:
                    prompt_text = f"{inputs}\n{self.cot_trigger}\n"
                messages.append({"role": "user", "content": prompt_text})
            elif isinstance(inputs, list):
                for msg in inputs:
                    if isinstance(msg, dict):
                        if "role" not in msg:
                            msg = {"role": "user", **msg}
                        messages.append(msg)
                    else:
                        messages.append({"role": "user", "content": str(msg)})
            elif isinstance(inputs, dict):
                if self.is_vision_model:
                    content = inputs.get("content", [])
                    processed_content = []
                    for item in content:
                        if isinstance(item, dict) and ("image_url" in item or "base64" in item):
                            processed_content.append(self._process_image_content(item))
                        else:
                            processed_content.append({"type": "text", "text": str(item)})
                    messages.append({"role": "user", "content": json.dumps(processed_content)})
                else:
                    messages.append({"role": "user", "content": str(inputs)})
            else:
                messages.append({"role": "user", "content": str(inputs)})
            payload["messages"] = messages
        else:
            # For the Completions API, use the prompt field.
            prompt_text = inputs if not (cot and self.cot_trigger) else f"{inputs}\n{self.cot_trigger}\n"
            payload["prompt"] = prompt_text
            payload["logprobs"] = params.get("logprobs") if return_logits else None
        
        # Set additional API parameters (e.g., max_tokens, temperature, top_p, etc.)
        for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in params:
                payload[param] = params[param]
        
        # Remove any keys with None values.
        return {k: v for k, v in payload.items() if v is not None}
    
    def _generate_single(
        self,
        input_item: Dict[str, Any],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates text for a single input item with retry logic and exponential backoff.
        
        Args:
            input_item (Dict[str, Any]): An input item containing at least the "input" key.
            return_logits (bool): Whether to return logits.
            until (Optional[Union[str, List[str]]]): Optional stop criteria.
            cot (bool): Whether to enable chain-of-thought prompting.
            **kwargs: Additional parameters to pass to the API.
        
        Returns:
            Dict[str, Any]: A dictionary with keys "prediction", "finish_reason", and optionally "logprobs".
        """
        result = None
        for attempt in range(self.max_retries):
            try:
                # Explicitly pass the 'cot' parameter to _create_payload.
                payload = self._create_payload(
                    input_item["input"],
                    return_logits=return_logits,
                    until=until,
                    cot=cot,
                    **kwargs,
                )
                if not self.use_chat_api:
                    # Call the traditional completions API.
                    response = openai.Completion.create(**payload)
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
                    # Call the Chat API.
                    response = openai.ChatCompletion.create(**payload)
                    result = {
                        "prediction": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits and hasattr(response.choices[0], "logprobs"):
                        result["logprobs"] = response.choices[0].logprobs
                break  # Exit the retry loop if successful.
            except Exception as e:
                if attempt == self.max_retries - 1:
                    error_msg = f"Error after {self.max_retries} attempts: {str(e)}"
                    raise RuntimeError(error_msg) from e
                else:
                    # Exponential backoff before retrying.
                    time.sleep(min(2 ** attempt, 32))
        return result
    
    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        use_chat_api: Optional[bool] = None,
        until: Optional[Union[str, List[str]]] = None,
        cot: bool = False,
        max_retries: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generates text for a batch of input items.
        
        If the model is a vision model, uses ThreadPoolExecutor for SDK-based calls.
        Otherwise, uses asynchronous httpx for improved performance.
        
        Args:
            inputs (List[Dict[str, Any]]): A list of input items. Each item must include the "input" key.
            return_logits (bool): If True, includes logits in the output (if supported).
            use_chat_api (Optional[bool]): Overrides the instance's use_chat_api flag if provided.
            until (Optional[Union[str, List[str]]]): Stop sequence(s) for generation.
            cot (bool): If True, enables chain-of-thought processing.
            max_retries (int | None): Maximum retry attempts for each API call.
            show_progress (bool): If True, displays a progress bar.
            **kwargs: Additional API parameters.
        
        Returns:
            List[Dict[str, Any]]: A list of input items updated with generation results.
        """
        if max_retries is not None:
            self.max_retries = max_retries
        
        # Override use_chat_api if provided
        if use_chat_api is not None:
            self.use_chat_api = use_chat_api
        
        logger.info(f"Starting batch generation for {len(inputs)} items.")
        
        # For vision models, continue to use the SDK client via ThreadPoolExecutor
        if self.is_vision_model:
            logger.info("Using ThreadPoolExecutor for vision model generation.")
            results = []
            max_workers = self.batch_size
            
            def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
                input_copy = deepcopy(item)
                result = self._generate_single(input_copy, return_logits, until, cot=cot, **kwargs)
                # If CoT is enabled and a parser is provided, apply CoT parsing to the generated text.
                if result and result.get("prediction") is not None and cot and self.cot_parser:
                    generated_text = result["prediction"]
                    cot_text, final_answer = self.cot_parser(generated_text)
                    result["chain_of_thought"] = cot_text
                    result["prediction"] = final_answer
                return result or {
                    "error": "Failed to generate",
                    "prediction": None,
                    "finish_reason": "error",
                }
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {executor.submit(process_item, item): item for item in inputs}
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Generating OpenAI outputs", disable=not show_progress):
                    orig_item = future_to_item[future]
                    try:
                        res = future.result()
                        merged = deepcopy(orig_item)
                        merged.update(res)
                        results.append(merged)
                    except Exception as e:
                        logger.error(f"Error in API call: {str(e)}")
                        merged = deepcopy(orig_item)
                        merged.update({
                            "error": str(e),
                            "prediction": None,
                            "finish_reason": "error"
                        })
                        results.append(merged)
        
        # For non-vision models, use httpx-based asynchronous calls
        else:
            logger.info("Using httpx for asynchronous generation.")
            
            # Apply nest_asyncio preemptively to avoid nested event loop issues
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass
                
            # Now run the async function with the configured event loop
            results = asyncio.run(
                self._generate_batch_httpx(inputs, return_logits, until, cot, **kwargs)
            )
        
        logger.info("Batch generation completed.")
        return results

    async def _send_single_request_httpx(
        self,
        client: httpx.AsyncClient,
        input_item: Dict[str, Any],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sends a single HTTP POST request using httpx.
        Implements retry logic with exponential backoff.
        """
        effective_retries = self.max_retries
        payload = self._create_payload(input_item["input"], return_logits=return_logits, until=until, cot=cot, **kwargs)
        attempt = 0
        
        # Determine the appropriate endpoint URL based on the API type
        api_url = f"{openai.api_base}/{'chat/completions' if self.use_chat_api else 'completions'}"
        
        # Prepare headers
        headers = {}
        if openai.api_key:
            headers["Authorization"] = f"Bearer {openai.api_key}"
        headers["Content-Type"] = "application/json"
        
        while attempt <= effective_retries:
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code} Error: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                resp_data = response.json()
                
                # Parse response based on API type
                if self.use_chat_api:
                    try:
                        content = resp_data["choices"][0]["message"]["content"]
                        finish_reason = resp_data["choices"][0].get("finish_reason", "")
                    except (KeyError, IndexError):
                        content = json.dumps(resp_data, indent=2)
                        finish_reason = ""
                else:
                    try:
                        content = resp_data["choices"][0]["text"]
                        finish_reason = resp_data["choices"][0].get("finish_reason", "")
                    except (KeyError, IndexError):
                        content = json.dumps(resp_data, indent=2)
                        finish_reason = ""
                
                result = {
                    "prediction": content,
                    "finish_reason": finish_reason,
                }
                
                # Add logprobs if requested and available
                if return_logits:
                    try:
                        if self.use_chat_api and "logprobs" in resp_data["choices"][0]:
                            result["logprobs"] = resp_data["choices"][0]["logprobs"]
                        elif not self.use_chat_api and "logprobs" in resp_data["choices"][0]:
                            result["logprobs"] = resp_data["choices"][0]["logprobs"]["token_logprobs"]
                            result["tokens"] = resp_data["choices"][0]["logprobs"]["tokens"]
                    except (KeyError, IndexError):
                        pass
                
                # Apply chain-of-thought parsing if enabled
                if cot and self.cot_parser and result["prediction"]:
                    generated_text = result["prediction"]
                    cot_text, final_answer = self.cot_parser(generated_text)
                    result["chain_of_thought"] = cot_text
                    result["prediction"] = final_answer
                
                return result
                
            except Exception as e:
                logger.error(f"HTTP attempt {attempt + 1}/{effective_retries} failed: {e}")
                attempt += 1
                # Add random jitter to backoff time to prevent thundering herd problem
                jitter = random.uniform(0, 1)
                backoff_time = min(2 ** attempt + jitter, 32)
                logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                await asyncio.sleep(backoff_time)
        
        raise RuntimeError(f"Failed after {effective_retries} retries via httpx.")

    async def _generate_batch_httpx(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously processes a batch of requests using httpx.
        Uses batching and rate limiting to avoid API rate limits.
        """
        logger.info(f"Starting asynchronous HTTP batch generation for {len(inputs)} items.")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 배치 크기 증가 - 더 많은 요청을 병렬로 처리
            batch_size = min(self.batch_size, 32)  # 16에서 32로 증가
            logger.info(f"Processing {len(inputs)} items in batches of {batch_size}")
            
            all_results = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(inputs) + batch_size - 1)//batch_size}")
                
                # Process current batch
                tasks = []
                for item in batch:
                    tasks.append(self._send_single_request_httpx(
                        client, item, return_logits, until, cot=cot, **kwargs
                    ))
                
                # Wait for all requests in this batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for orig, res in zip(batch, batch_results):
                    merged = deepcopy(orig)
                    if isinstance(res, Exception):
                        logger.error(f"Error processing request: {str(res)}")
                        merged.update({
                            "error": str(res),
                            "prediction": None,
                            "finish_reason": "error"
                        })
                    else:
                        merged.update(res)
                    all_results.append(merged)
                
                # 배치 간 대기 시간 최소화
                if i + batch_size < len(inputs):
                    logger.info("Waiting 0.2 seconds before next batch...")
                    await asyncio.sleep(0.2)  # 0.5초에서 0.2초로 감소
            
            logger.info("Asynchronous HTTP batch generation completed.")
            return all_results