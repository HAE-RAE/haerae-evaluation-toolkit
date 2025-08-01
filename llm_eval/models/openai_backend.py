import asyncio
import json
import logging
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
from tqdm import tqdm

from llm_eval.utils.logging import get_logger

from . import register_model
from .base import BaseModel

# Create a logger instance for this module
logger = get_logger(name="openai_backend", level=logging.INFO)


@register_model("openai")
class OpenAIModel(BaseModel):
    """
    OpenAIModel implements a production-grade backend that supports both
    vision and text models using the official OpenAI Python client.

    All requests are handled asynchronously via ``openai.AsyncOpenAI`` so
    that multiple prompts can be processed concurrently (controlled by
    ``batch_size``).

    Key Features:
      - Constructs payloads for both Chat and Completions API calls.
      - Supports chain-of-thought (CoT) prompting and parsing.
      - Implements robust retry logic with exponential backoff.
      - Uses asyncio to concurrently process a batch of requests.

    Args:
        api_key (Optional[str]): OpenAI API key (optional if using an OpenAI-compatible server).
        api_base (str): Base URL for the API.
        model_name (str): Identifier of the model (e.g., "gpt-4", "gpt-3.5-turbo", "Qwen/Qwen2.5-7B-Instruct").
        system_message (Optional[str]): System message to include for Chat API.
        use_chat_api (bool): Flag to determine whether to use Chat API (True) or Completions API (False).
        is_vision_model (bool): Whether the model is a vision model.
        cot_trigger (Optional[str]): A trigger phrase for chain-of-thought prompting.
        cot_parser (Optional[Callable[[str], Tuple[str, str]]]): Function to parse generated text into (chain_of_thought, final_answer).
        batch_size (int): Number of concurrent requests to send.
        max_retries (int): Maximum number of retry attempts for API calls.
        timeout (Optional[float]): Timeout (in seconds) for HTTP requests.
        cot (bool): Flag to enable chain-of-thought prompting.
        **kwargs: Additional API parameters (e.g., temperature, max_tokens, top_p, etc.).
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
        timeout: Optional[float] = 30.0,
        cot: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not model_name or not api_base:
            raise ValueError("model_name and api_base are required")

        self.is_vision_model = is_vision_model
        self.use_chat_api = use_chat_api
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.cot = cot

        self.model_name = model_name
        self.system_message = system_message
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser  # Function to parse CoT responses, if enabled
        # Additional parameters such as temperature, max_tokens, etc.
        self.default_params = kwargs

        # Single OpenAI client (async) used for both text and vision models
        self.api_base = api_base
        if api_key:
            self.api_key = api_key
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=self.timeout,
            max_retries=0,
        )
        logger.info("Using OpenAI Async client for generation.")

    def _process_image_content(self, content: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Processes image content into the required format for the OpenAI Vision API.
        Supports URLs, base64 strings, or dictionaries with detailed specifications.

        Returns:
            Dict[str, Any]: Processed image information.
        """
        # Implementation omitted for brevity.
        pass

    def _create_payload(
        self,
        inputs: Union[str, List[Dict], Dict],
        cot: bool = False,
        until: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs the API payload for a call.

        If using the Chat API, constructs a messages list with an optional system message.
        If CoT is enabled, appends the CoT trigger to the prompt.
        If 'until' is provided, adds it as a stop sequence.

        Args:
            inputs: The input prompt (string or pre-constructed list/dict for messages).
            cot (bool): Whether to enable chain-of-thought prompting.
            until (Optional[Union[str, List[str]]]): Stop sequence(s) for generation.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, Any]: The API payload dictionary.
        """
        params = deepcopy(self.default_params)
        params.update(kwargs)

        payload = {}
        if self.use_chat_api:
            messages = []
            if self.system_message:
                messages.append(
                    {"role": "system", "content": self.system_message})
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
                if isinstance(until, str):
                    until = [until]
                payload["stop"] = until
        else:
            prompt_text = inputs if not (
                cot and self.cot_trigger) else f"{inputs}\n{self.cot_trigger}\n"
            payload = {"model": self.model_name, "prompt": prompt_text}
            if params.get("logprobs") is not None:
                payload["logprobs"] = params["logprobs"]
            if until is not None:
                if isinstance(until, str):
                    until = [until]
                payload["stop"] = until

        # Add common parameters (if provided) such as max_tokens, temperature, etc.
        for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in params:
                payload[param] = params[param]

        # Remove any keys with None values
        return {k: v for k, v in payload.items() if v is not None}

    def _execute_tool_calls(self, tool_calls: List[dict]) -> str:
        """
        Executes tool calls if present in the response.
        For production, this should invoke the corresponding functions; here, it simply concatenates the tool names.

        Args:
            tool_calls: List of tool call dictionaries.

        Returns:
            str: Concatenated string indicating executed tool calls.
        """
        return "\n".join([f"Executed tool: {tc.get('function', {}).get('name', 'unknown')}" for tc in tool_calls])

    def _parse_normal_response(self, resp_data: dict) -> str:
        """
        Parses a non-streaming API response.
        Expects the response in OpenAI ChatCompletion format.
        If tool_calls are present, executes them.

        Args:
            resp_data: The JSON response from the API.

        Returns:
            str: The extracted content or a formatted JSON string on failure.
        """
        try:
            message = resp_data["choices"][0]["message"]
            if "tool_calls" in message and message["tool_calls"]:
                return self._execute_tool_calls(message["tool_calls"])
            return message.get("content", json.dumps(resp_data, indent=2))
        except (KeyError, IndexError):
            return json.dumps(resp_data, indent=2)


    async def _send_single_request_async(
        self,
        client: openai.AsyncOpenAI,
        item: Dict[str, Any],
        return_logits: bool,
        until: Optional[Union[str, List[str]]],
        cot: bool = False,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a single request using the OpenAI async client with retries."""
        effective_retries = max_retries if max_retries is not None else self.max_retries
        payload = self._create_payload(
            item["input"], cot=cot, until=until, **kwargs)

        attempt = 0
        while attempt <= effective_retries:
            try:
                if self.use_chat_api:
                    response = await client.chat.completions.create(**payload)
                    result = {
                        "prediction": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits and hasattr(response.choices[0], "logprobs"):
                        result["logprobs"] = response.choices[0].logprobs
                else:
                    response = await client.completions.create(**payload)
                    result = {
                        "prediction": response.choices[0].text,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                    if return_logits and hasattr(response.choices[0], "logprobs"):
                        result.update({
                            "logprobs": response.choices[0].logprobs.token_logprobs,
                            "tokens": response.choices[0].logprobs.tokens,
                        })
                if cot and self.cot_parser:
                    generated_text = result["prediction"]
                    cot_text, final_answer = self.cot_parser(generated_text)
                    result["chain_of_thought"] = cot_text
                    result["prediction"] = final_answer
                return result
            except Exception as e:
                logger.error(
                    f"OpenAI attempt {attempt + 1}/{effective_retries} failed: {e}")
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, 32))
        raise RuntimeError(
            f"Failed after {effective_retries} retries via OpenAI client.")


    async def _generate_batch_async(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        until: Optional[Union[str, List[str]]] = None,
        cot: bool = False,
        max_retries: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Internal async helper for batch generation."""
        logger.info(f"Starting batch generation for {len(inputs)} items.")
        results: List[Dict[str, Any]] = []
        async with self.client as client:
            tasks = [
                self._send_single_request_async(
                    client,
                    item,
                    return_logits,
                    until,
                    cot=cot,
                    max_retries=max_retries,
                    **kwargs,
                )
                for item in inputs
            ]
            for item, task in zip(
                inputs,
                tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Generating outputs",
                    disable=not show_progress,
                ),
            ):
                try:
                    res = await task
                    merged = deepcopy(item)
                    merged.update(res)
                    results.append(merged)
                except Exception as e:
                    logger.error(f"OpenAI error: {str(e)}")
                    error_item = deepcopy(item)
                    error_item.update({
                        "error": str(e),
                        "prediction": None,
                        "finish_reason": "error",
                    })
                    results.append(error_item)
        logger.info("Batch generation completed.")
        return results

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
        """Public API for batch generation using asyncio."""
        return asyncio.run(
            self._generate_batch_async(
                inputs,
                return_logits=return_logits,
                until=until,
                cot=cot,
                max_retries=max_retries,
                show_progress=show_progress,
                **kwargs,
            )
        )
