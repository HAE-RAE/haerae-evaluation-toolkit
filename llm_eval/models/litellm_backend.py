import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import litellm
from tqdm import tqdm

from . import register_model
from .base import BaseModel
from llm_eval.utils.prompt_template import default_cot_parser  
from llm_eval.utils.logging import get_logger

logger = get_logger(name="litellm", level=logging.INFO)

@register_model("litellm")
class LiteLLMBackend(BaseModel):
    """
    A backend model class that uses LiteLLM.
    
    Args:
        provider (str): Name of the LLM provider (e.g., "openai", "anthropic", "bedrock", "azure").
        model_name (str): The model name to use.
        api_key (Optional[str]): API key for providers like OpenAI or Azure.
        api_base (Optional[str]): Base URL for the API.
        aws_access_key_id (Optional[str]): AWS access key ID (for Bedrock).
        aws_secret_access_key (Optional[str]): AWS secret key (for Bedrock).
        anthropic_api_key (Optional[str]): Anthropic API key.
        max_new_tokens (int): Maximum number of tokens to generate per call.
        temperature (float): Sampling temperature (0.0 ~ 1.0).
        batch_size (int): Number of items to process concurrently (using multithreading).
        cot (bool): If True, appends a CoT trigger to the prompt.
        cot_trigger (Optional[str]): Trigger phrase for inducing CoT. If None, no trigger is used.
        cot_parser (Optional[Callable[[str], Tuple[str, str]]]): A callable to parse generated text into 
            (chain_of_thought, final_answer). Defaults to default_cot_parser.
        **kwargs: Additional configuration parameters.
    """
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        batch_size: int = 8,
        is_vision_model: bool = False,
        cot: bool = False,
        cot_trigger: Optional[str] = None,
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = default_cot_parser,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"[LiteLLMBackend] Loading model settings for '{model_name}' from provider '{provider}'.")
        
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.is_vision_model = is_vision_model
        self.cot = cot
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser
        self.extra_kwargs = kwargs

        # Configure API settings specific to LiteLLM (align with latest docs)
        self.completion_kwargs = {}
        if api_key is not None:
            self.completion_kwargs["api_key"] = api_key
        if api_base is not None:
            self.completion_kwargs["api_base"] = api_base
        if api_version is not None:
            self.completion_kwargs["api_version"] = api_version
        if self.provider == "bedrock":
            # bedrock uses model identifiers like "bedrock/<model>" and reads AWS params if provided
            for k, v in {
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            }.items():
                if v is not None:
                    self.completion_kwargs[k] = v
        elif self.provider == "anthropic" and anthropic_api_key is not None:
            self.completion_kwargs["api_key"] = anthropic_api_key

    def _process_image_for_litellm(self, image_input: Union[str, Dict]) -> Dict[str, Any]:
        """
        Process image input for LiteLLM format.
        
        Args:
            image_input: Image URL, file path, or dict with image info
            
        Returns:
            Dict formatted for LiteLLM vision models
        """
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # URL
                return {
                    "type": "image_url",
                    "image_url": {"url": image_input}
                }
            elif image_input.startswith('data:image/'):
                # Base64 data URL
                return {
                    "type": "image_url",
                    "image_url": {"url": image_input}
                }
            else:
                # File path - convert to base64
                import base64
                import mimetypes
                try:
                    with open(image_input, 'rb') as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        mime_type = mimetypes.guess_type(image_input)[0] or 'image/jpeg'
                        return {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                        }
                except Exception as e:
                    logger.error(f"Failed to process image file {image_input}: {e}")
                    return {
                        "type": "text",
                        "text": f"[Error processing image: {image_input}]"
                    }
        elif isinstance(image_input, dict):
            # Already in correct format or needs adjustment
            if "type" in image_input:
                return image_input
            elif "url" in image_input:
                return {
                    "type": "image_url",
                    "image_url": {"url": image_input["url"]}
                }
            else:
                logger.warning(f"Unknown image format: {image_input}")
                return {
                    "type": "text",
                    "text": f"[Unknown image format: {image_input}]"
                }
        else:
            logger.warning(f"Unsupported image type: {type(image_input)}")
            return {
                "type": "text",
                "text": f"[Unsupported image type: {type(image_input)}]"
            }

    def _prepare_completion_kwargs(self, prompt: str, images: Optional[List] = None, until: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Prepares the parameters to pass to the LiteLLM completion function.
        
        Args:
            prompt (str): The input prompt text.
            images (Optional[List]): Optional list of images for vision models.
            until (Optional[Union[str, List[str]]]): Optional stop sequence(s).
        
        Returns:
            Dict[str, Any]: Dictionary of completion parameters.
        """
        # Append the CoT trigger if chain-of-thought is enabled.
        if self.cot and self.cot_trigger:
            prompt = f"{prompt}\n{self.cot_trigger}"
        
        # Handle vision inputs
        if images and self.is_vision_model:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append(self._process_image_for_litellm(image))
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]
            
        completion_kwargs = {
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **self.completion_kwargs,
            **self.extra_kwargs,
        }
        if until is not None:
            completion_kwargs["stop"] = until if isinstance(until, list) else [until]
        
        # Adjust model name formatting based on provider (latest LiteLLM uses prefixes like 'azure/<deployment>')
        if self.provider in {"azure", "bedrock"}:
            completion_kwargs["model"] = f"{self.provider}/{self.model_name}"
        else:
            completion_kwargs["model"] = self.model_name
        
        logger.debug(f"[LiteLLMBackend] Prepared completion kwargs: {completion_kwargs}")
        return completion_kwargs

    async def _generate_once_async(self, completion_kwargs: Dict[str, Any]) -> str:
        resp = await litellm.acompletion(**completion_kwargs)
        return resp.choices[0].message.content

    async def _generate_with_retry_async(
        self,
        completion_kwargs: Dict[str, Any],
        max_attempts: int = 3,
        initial_wait: float = 1.0,
    ) -> str:
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < max_attempts:
            try:
                return await self._generate_once_async(completion_kwargs)
            except Exception as e:
                last_exception = e
                attempt += 1
                if attempt < max_attempts:
                    wait_time = initial_wait * (2 ** (attempt - 1))
                    logger.warning(
                        f"[LiteLLMBackend] Attempt {attempt} failed: {e}. Retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
        error_msg = f"[LiteLLMBackend] All {max_attempts} attempts failed. Last error: {last_exception}"
        logger.error(error_msg)
        raise last_exception or Exception(error_msg)

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        batch_size: Optional[Union[int, str]] = None,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Public API. Runs async generation with asyncio for concurrency."""
        # Optional runtime override of batch_size
        if batch_size is not None:
            if isinstance(batch_size, str) and batch_size.lower() == "auto":
                self.batch_size = min(len(inputs), 8)
                logger.info(f"[LiteLLMBackend] Using auto batch size: {self.batch_size}")
            elif isinstance(batch_size, int) and batch_size > 0:
                self.batch_size = batch_size

        return asyncio.run(
            self._generate_batch_async(
                inputs,
                return_logits=return_logits,
                until=until,
                show_progress=show_progress,
                **kwargs,
            )
        )

    async def _generate_batch_async(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if return_logits:
            raise NotImplementedError("LiteLLM backend does not support logits calculation yet.")

        semaphore = asyncio.Semaphore(self.batch_size if isinstance(self.batch_size, int) else 8)

        async def _worker(item: Dict[str, Any]) -> Dict[str, Any]:
            # Extract text and images from the item
            text_input, images = self._extract_text_and_images(item)
            reference = item.get("reference", "")
            completion_kwargs = self._prepare_completion_kwargs(text_input, images=images, until=until)
            async with semaphore:
                try:
                    prediction = await self._generate_with_retry_async(
                        completion_kwargs,
                        max_attempts=getattr(self, "retry_max", 3),
                        initial_wait=getattr(self, "retry_base_delay", 1.0),
                    )
                    result_item = {
                        "input": item.get("input", ""),
                        "reference": reference,
                        "prediction": prediction,
                    }
                    if self.cot and self.cot_parser:
                        try:
                            chain_of_thought, final_answer = self.cot_parser(prediction)
                            result_item["chain_of_thought"] = chain_of_thought
                            result_item["prediction"] = final_answer
                        except Exception as e:
                            logger.warning(f"[LiteLLMBackend] CoT parsing failed: {e}")
                    return result_item
                except Exception as e:
                    logger.error(f"[LiteLLMBackend] Error generating completion: {str(e)}")
                    return {
                        "input": prompt,
                        "reference": reference,
                        "prediction": f"Error: {str(e)}",
                    }

        tasks = [_worker(item) for item in inputs]
        results: List[Dict[str, Any]] = []
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LiteLLM Batch Generation", disable=not show_progress):
            try:
                results.append(await fut)
            except Exception as e:
                logger.error(f"[LiteLLMBackend] Task failed: {e}")
                results.append({"input": None, "reference": None, "prediction": f"Error: {e}"})
        return results
