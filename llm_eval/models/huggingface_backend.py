import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

try:
    from transformers import AutoProcessor
    from PIL import Image
    import requests
    VISION_AVAILABLE = True
except ImportError:
    AutoProcessor = None
    Image = None
    requests = None
    VISION_AVAILABLE = False

from .base import BaseModel 
from . import register_model
from llm_eval.utils.logging import get_logger 
from llm_eval.utils.prompt_template import default_cot_parser 

logger = get_logger(name="huggingface", level=logging.INFO)


@register_model("huggingface")
class HuggingFaceModel(BaseModel):
    """
    A backend model class that uses HuggingFace Transformers.

    Main points:
      - If `return_logits=True`, we call `model.generate(..., return_dict_in_generate=True, output_scores=True)`
        so that "scores" (logits per step) are included in the output, allowing us to compute log probabilities.
      - If a `cot_parser` function is provided, we can separate chain-of-thought (CoT) text from the final answer.
      - If an input contains an "options" field (MCQA mode), the log likelihood for each option is computed.
        The option with the highest log probability is selected as the final prediction.
      - If `device="map"`, uses `device_map="auto"` for distributed model loading.
      - Supports vision-language models when `is_vision_model=True` is set.
    
    Args:
        model_name_or_path (str): HF Hub model ID or local path.
        tokenizer_id_or_path (str|None): Optional tokenizer ID or local path. If None, uses model_name_or_path.
        dtype (str): 'auto', 'fp16', 'bf16', 'fp32', etc. If 'auto', uses the model's native precision.
        max_new_tokens (int): Maximum new tokens to generate in one call.
        max_input_tokens (int|None): Optional maximum input tokens. If None, no truncation.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        stop_token (str|None): Optional stop token. If None, no stopping.
        device (str|None): 'cpu', 'cuda', 'cuda:0', 'map', etc. If None, uses the model's device.
                         If "map", uses device_map="auto".
        is_vision_model (bool): Whether this is a vision-language model.
        cot (bool): Whether to use chain-of-thought prompting.
        cot_trigger (str|None): Optional CoT (Chain-of-Thought) trigger appended to the prompt. If None, no CoT trigger.
        cot_parser (callable|None): A function that takes a string (generated text) and returns a tuple 
                                  (chain_of_thought, final_answer). If None, no CoT parsing is applied.
        **kwargs: Additional parameters as needed.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_id_or_path: Optional[str] = None,
        dtype: str = "auto",
        max_new_tokens: int = 512,
        max_input_tokens: Optional[int] = None, # Not directly used in this snippet but kept for consistency
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        stop_token: Optional[str] = None, # Not used in this snippet but kept for consistency
        device: Optional[str] = None,
        batch_size: int = 1,
        is_vision_model: bool = False,
        cot: bool = False,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = default_cot_parser,
        **kwargs
    ):
        # Call parent class constructor to initialize base attributes
        super().__init__(**kwargs)
        
        # Extract model name from model ID
        self.model_name = f"huggingface:{model_name_or_path}"
        self.is_vision_model = is_vision_model
        logger.info(f"[HuggingFaceModel] Initializing with model: {model_name_or_path}, vision_model: {is_vision_model}")

        # Check vision dependencies
        if self.is_vision_model and not VISION_AVAILABLE:
            raise ImportError("Vision model requested but required dependencies (PIL, requests) are not available. "
                            "Please install them with: pip install Pillow requests")

        # Dtype setup
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:
            logger.warning(f"Unsupported dtype '{dtype}', defaulting to 'auto'.")
            torch_dtype = "auto"
        
        self.device_arg = device # Store the original device argument
        self.is_model_distributed = False

        # Load tokenizer/processor
        _tokenizer_path = tokenizer_id_or_path if tokenizer_id_or_path else model_name_or_path
        
        if self.is_vision_model:
            # For vision models, try to load processor first, fallback to tokenizer
            try:
                self.processor = AutoProcessor.from_pretrained(_tokenizer_path)
                self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
                logger.info(f"[HuggingFaceModel] Loaded processor for vision model")
            except Exception as e:
                logger.warning(f"Failed to load processor, falling back to tokenizer: {e}")
                self.processor = None
                self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path, padding_side="left")
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path, padding_side="left")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"[HuggingFaceModel] Tokenizer pad_token set to eos_token: {self.tokenizer.eos_token}")

        # Load model
        logger.info(f"[HuggingFaceModel] Loading model from {model_name_or_path} with dtype: {torch_dtype}")
        if device == "map":
            logger.info("[HuggingFaceModel] Using device_map='auto' for model loading.")
            if self.is_vision_model:
                # For vision models, we might need different model classes
                try:
                    from transformers import AutoModelForVision2Seq
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_name_or_path,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        **kwargs.get("model_kwargs", {})
                    )
                except ImportError:
                    logger.warning("AutoModelForVision2Seq not available, using AutoModelForCausalLM")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        **kwargs.get("model_kwargs", {})
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    **kwargs.get("model_kwargs", {}) # Pass other model specific kwargs
                )
            self.is_model_distributed = True
            # For distributed models, input tensors are typically kept on CPU.
            # The model's internal logic handles moving parts of the computation to appropriate devices.
            # self.device will refer to the device of the first parameter, but we don't move all inputs there.
            self.device = self.model.device # This might be e.g., 'cuda:0' for the first shard
            logger.info(f"[HuggingFaceModel] Model distributed. Main device (first shard): {self.device}")
        else:
            if self.is_vision_model:
                try:
                    from transformers import AutoModelForVision2Seq
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch_dtype,
                        **kwargs.get("model_kwargs", {})
                    )
                except ImportError:
                    logger.warning("AutoModelForVision2Seq not available, using AutoModelForCausalLM")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch_dtype,
                        **kwargs.get("model_kwargs", {})
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    **kwargs.get("model_kwargs", {}) # Pass other model specific kwargs
                )
            self.device = device
            if self.device:
                logger.info(f"[HuggingFaceModel] Moving model to device: {self.device}")
                try:
                    self.model.to(self.device)
                except Exception as e:
                    logger.error(f"Failed to move model to {self.device}: {e}. Attempting to load on CPU.")
                    self.device = "cpu"
                    self.model.to(self.device) # Fallback to CPU
            else: # device is None
                self.device = self.model.device # Get the device it was loaded on
                logger.info(f"[HuggingFaceModel] Model device not specified, loaded on: {self.device}")
        
        self.model.eval()

        # Inference hyperparameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.cot = cot
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser
        self.batch_size = batch_size

    def _load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        Load an image from various input formats.
        
        Args:
            image_input: Can be a file path, URL, or PIL Image
            
        Returns:
            PIL Image object
        """
        if not VISION_AVAILABLE:
            raise ImportError("Vision dependencies not available")
            
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # URL
                try:
                    response = requests.get(image_input)
                    response.raise_for_status()
                    return Image.open(requests.get(image_input, stream=True).raw)
                except Exception as e:
                    logger.error(f"Failed to load image from URL {image_input}: {e}")
                    raise
            else:
                # File path
                try:
                    return Image.open(image_input)
                except Exception as e:
                    logger.error(f"Failed to load image from file {image_input}: {e}")
                    raise
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _prepare_vision_inputs(self, text: str, images: List, **kwargs) -> Dict[str, Any]:
        """
        Prepare inputs for vision-language models.
        
        Args:
            text: Text prompt
            images: List of images
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of prepared inputs
        """
        if not self.is_vision_model or not images:
            # Fallback to text-only processing
            return self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else 2048
            )
        
        # Load images
        pil_images = []
        for img in images:
            try:
                pil_images.append(self._load_image(img))
            except Exception as e:
                logger.warning(f"Failed to load image {img}: {e}, skipping")
                continue
        
        if not pil_images:
            logger.warning("No valid images found, falling back to text-only")
            return self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else 2048
            )
        
        # Use processor if available
        if self.processor:
            try:
                return self.processor(
                    text=text,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            except Exception as e:
                logger.error(f"Processor failed: {e}, falling back to tokenizer")
                return self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else 2048
                )
        else:
            # No processor available, use tokenizer only
            logger.warning("No processor available for vision model, using text-only tokenizer")
            return self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else 2048
            )

    def _score_option(self, prompt: str, option: str) -> float:
        """
        Computes the log likelihood score for a given prompt-option pair.
        Concatenates the prompt and option, tokenizes the combined text, and sums the log probabilities
        for the tokens corresponding to the option.
        """
        full_text = prompt + " " + option
        # Tokenizer by default puts tensors on CPU
        encoded_full = self.tokenizer(full_text, return_tensors="pt")
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = encoded_prompt.input_ids.shape[1]
        
        input_ids = encoded_full.input_ids
        attention_mask = encoded_full.attention_mask

        # Move tensors to the appropriate device if not distributed
        # If distributed (device_map="auto"), inputs should stay on CPU for the model call.
        # The model's forward pass handles distributing data internally.
        if not self.is_model_distributed and self.device and self.device != "cpu":
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        # Else, if distributed or on CPU, keep on CPU.

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
        
        # Ensure logits are on CPU for subsequent operations if they were on GPU
        # This is important if the model is on a single GPU and we need to process logits on CPU
        # For distributed models, logits for specific parts might be on different devices.
        # However, .logits from the forward pass of the whole model should be gatherable or on a primary device.
        # For simplicity here, we assume we can bring them to CPU if needed.
        # Usually for log_softmax and indexing, it's fine to keep them on the device where they are.
        # log_probs = F.log_softmax(logits.cpu() if not self.is_model_distributed and self.device and self.device != "cpu" else logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)


        option_ids = input_ids[0, prompt_len:] # Still on the original device (CPU or self.device)
        
        total_log_prob = 0.0
        for t, token_id in enumerate(option_ids):
            # Ensure indexing happens on the correct device where log_probs and token_id reside
            # The log_probs tensor is (1, seq_len, vocab_size)
            # We need log_probs[0, index_of_token_before_current_one, current_token_id]
            # The logits are for predicting the *next* token. So, for token at full_text[k],
            # we use logits from position full_text[k-1].
            # Option tokens start at index prompt_len in input_ids.
            # The log_prob for input_ids[0, prompt_len] is taken from logits[0, prompt_len-1, :]
            # The log_prob for input_ids[0, prompt_len+t] is taken from logits[0, prompt_len+t-1, :]
            
            logit_idx = prompt_len + t -1 
            if logit_idx < 0 or logit_idx >= log_probs.shape[1]: # Should not happen with proper inputs
                logger.warning(f"Logit index {logit_idx} out of bounds for log_probs shape {log_probs.shape}. Skipping token.")
                continue

            # Ensure token_id is scalar for indexing
            current_token_id = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
            
            try:
                # Indexing: log_probs[batch_idx, sequence_idx, token_idx]
                token_log_prob = log_probs[0, logit_idx, current_token_id].item()
                total_log_prob += token_log_prob
            except IndexError as e:
                logger.error(f"Error indexing log_probs: {e}. log_probs shape: {log_probs.shape}, logit_idx: {logit_idx}, token_id: {current_token_id}")
                # Potentially add a very small number or handle error appropriately
                total_log_prob -= 1e9 # Penalize heavily if indexing fails

        return total_log_prob

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = True, # Note: this return_logits is for generate, not _score_option
        batch_size: Optional[Union[int, str]] = None,
        until: Optional[Union[str, List[str]]] = None,
        show_progress: bool = True,
        **kwargs # kwargs for model.generate
    ) -> List[Dict[str, Any]]:
        if batch_size is None:
            batch_size = self.batch_size
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            auto_mode = True
            # For distributed models, a large CPU-side batch might still be fine,
            # but actual GPU memory per device is the constraint.
            # Transformers `generate` with `device_map` handles internal batching if any.
            # This batch_size is more about how many items we process in one Python loop iteration.
            current_bs = 128 if not self.is_model_distributed else len(inputs) # Process all if distributed, let HF handle micro-batching
            logger.info(f"[HuggingFaceModel] Batch size set to 'auto'. Starting with batch size {current_bs}.")
        else:
            auto_mode = False
            current_bs = batch_size if isinstance(batch_size, int) else len(inputs)
            logger.info(f"[HuggingFaceModel] Batch size set to {current_bs}.")

        stopping_criteria = None
        if until is not None:
            if isinstance(until, str):
                until = [until]

            class StoppingCriteriaSub(StoppingCriteria):
                def __init__(self, tokenizer, stops: List[str]):
                    super().__init__()
                    self.tokenizer = tokenizer
                    self.stops = stops
                    # Encode stop sequences. Handle potential issues if stop sequences are not in vocab.
                    # For simplicity, we'll decode generated text and check.
                    # More robust: use token IDs of stop sequences.
                
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    # input_ids is the entire sequence generated so far for the current batch item
                    # We only check the first item in the batch for stopping if batch_size > 1 in generate
                    # This is a simplification; true per-sequence stopping needs more complex handling
                    # or reliance on features in HF `generate`.
                    if input_ids.shape[0] > 1 : # if batched generation
                        # This stopping criteria might stop all if one stops.
                        # For fine-grained control, batch size 1 for generate or more advanced criteria.
                        # For now, assume we check the first sequence.
                         decoded_texts = self.tokenizer.batch_decode(input_ids)
                         return any(any(stop in decoded for stop in self.stops) for decoded in decoded_texts)
                    else: # single item generation
                        decoded = self.tokenizer.decode(input_ids[0])
                        return any(stop in decoded for stop in self.stops)


            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(self.tokenizer, stops=until)])
            if "stopping_criteria" not in kwargs: # Allow overriding from kwargs
                 kwargs["stopping_criteria"] = stopping_criteria

        results_final = []
        while True: # Loop for auto batch size reduction
            try:
                results_temp_batch = []
                # Process inputs in chunks.
                for start in tqdm(range(0, len(inputs), current_bs), disable=not show_progress, desc="Processing batches"):
                    batch_items = inputs[start:start + current_bs]
                    processed_items_in_batch = []

                    # Separate MCQA items from normal generation items
                    mcqa_items_indices = []
                    normal_items_indices = []
                    
                    current_prompts_for_normal_gen = []
                    original_normal_items = []

                    for idx, item in enumerate(batch_items):
                        is_mcqa = "options" in item and isinstance(item["options"], list)
                        if is_mcqa:
                            mcqa_items_indices.append(idx)
                        else:
                            normal_items_indices.append(idx)
                            prompt = f"{item['input']}\n{self.cot_trigger}\n" if (self.cot and self.cot_trigger) else item["input"]
                            current_prompts_for_normal_gen.append(prompt)
                            original_normal_items.append(item)
                    
                    # Process MCQA items individually (as _score_option is not batched here)
                    for item_idx_in_batch in mcqa_items_indices:
                        item = batch_items[item_idx_in_batch]
                        prompt = item.get("input", "")
                        option_log_probs = []
                        for opt_idx, opt in enumerate(item["options"]):
                            lp = self._score_option(prompt, opt)
                            option_log_probs.append(lp)
                            logger.debug(f"MCQA Option '{opt}' (idx {opt_idx}) for input '{prompt[:50]}...' -> log_prob: {lp}")
                        
                        if return_logits: # This refers to the generate_batch's return_logits
                            item.setdefault("logits", {})["option_log_probs"] = option_log_probs
                        
                        if not option_log_probs: # Should not happen if options exist
                             logger.error(f"No option log_probs for item: {item.get('id', 'N/A')}")
                             item["prediction"] = "Error: No options scored"
                        else:
                            best_idx = option_log_probs.index(max(option_log_probs))
                            item["prediction"] = item["options"][best_idx]
                        
                        # CoT parsing for MCQA (if applicable, though less common for pure scoring)
                        if self.cot and self.cot_parser: # Apply CoT parsing even if prediction is from scoring
                            # For now, let's assume no CoT text is generated *by scoring*, so chain_of_thought is None for MCQA.
                            item["chain_of_thought"] = None # Typically, _score_option doesn't produce CoT text.
                            # If a CoT string should be *added* even for MCQA, that logic would go here.
                            # E.g., if cot_trigger itself is considered the CoT.
                            if self.cot_trigger and item.get("prediction"): # Minimal CoT if trigger exists
                                 # This is a placeholder; real CoT for MCQA would be more involved.
                                 # Let's assume for now that if CoT is enabled, and we are in MCQA,
                                 # the prediction might still need to be parsed by cot_parser if it expects a specific format.
                                 # However, default_cot_parser might not be suitable if the prediction is just an option.
                                 # Revisit this CoT application for MCQA based on specific needs.
                                 # For now, if a cot_parser is given, we try to parse, assuming it can handle plain answers.
                                 generated_text_for_parser = item["prediction"]
                                 if self.cot_trigger : # Prepend a conceptual CoT if trigger exists
                                     generated_text_for_parser = f"{self.cot_trigger}\n{item['prediction']}" # This is a bit artificial for MCQA
                                 
                                 # Only parse if parser is available
                                 if self.cot_parser:
                                    try:
                                        # The parser might expect a full generation including the prompt.
                                        # For MCQA, this is tricky. Let's try parsing the prediction as is.
                                        # A more robust way might be to generate a rationale for the chosen option.
                                        # Assuming the cot_parser can take just the answer part:
                                        cot_text, final_answer_from_parser = self.cot_parser(item["prediction"]) # Try parsing just the selected option
                                        if cot_text: # If parser found CoT markers
                                             item["chain_of_thought"] = cot_text
                                             item["prediction"] = final_answer_from_parser
                                    except Exception as e_parse:
                                        logger.debug(f"CoT parser failed for MCQA item prediction '{item['prediction']}': {e_parse}. Using prediction as is.")
                                        # Keep original prediction if parsing fails or doesn't apply cleanly
                            
                        processed_items_in_batch.append(item)


                    # Process normal generation items in a batch
                    if current_prompts_for_normal_gen:
                        generated_outputs_normal = self._generate_normal(
                            original_normal_items, # Pass original items to map back
                            current_prompts_for_normal_gen,
                            return_logits=return_logits, # generate_batch's return_logits
                            stopping_criteria=kwargs.get("stopping_criteria"), # from above or user
                            **{k:v for k,v in kwargs.items() if k != "stopping_criteria"} # other gen kwargs
                        )
                        processed_items_in_batch.extend(generated_outputs_normal)
                    
                    # The processed_items_in_batch might not be in original order if MCQA and normal items were mixed.
                    # We need to re-sort them or ensure they are added correctly.
                    # For simplicity, let's assume the current loop processes one batch_item at a time if mcqa is involved
                    # or one batch of normal items.
                    # The current logic processes MCQA items then normal items. This is fine.
                    results_temp_batch.extend(processed_items_in_batch)
                
                results_final.extend(results_temp_batch) # Add successfully processed batch results
                break # Successfully processed all inputs

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if auto_mode:
                        if current_bs > 1:
                            current_bs = max(1, current_bs // 2)
                            logger.warning(f"[HuggingFaceModel] OOM detected: reducing batch size to {current_bs}.")
                            results_final.clear() # Clear partial results from the failed batch size
                            results_temp_batch.clear()
                            # Reset loop to retry with smaller batch_size for all inputs
                        else:
                            logger.error("[HuggingFaceModel] Batch size is 1 but OOM still occurs.")
                            raise RuntimeError("Out of memory even with batch size=1.") from e
                    else: # Not auto_mode
                        logger.error(f"[HuggingFaceModel] OOM with the specified batch size {current_bs}.")
                        raise RuntimeError(f"Out of memory with the specified batch size {current_bs}.") from e
                else: # Other RuntimeError
                    logger.error("[HuggingFaceModel] RuntimeError occurred:", exc_info=True)
                    raise
            except Exception as e_global: # Catch any other exception
                logger.error(f"[HuggingFaceModel] An unexpected error occurred: {e_global}", exc_info=True)
                raise

        return results_final


    def _generate_normal(
        self, 
        batch_items_original: List[Dict[str, Any]], # Original items for result mapping
        prompts: List[str], # Prompts already pre-processed (e.g. with CoT trigger)
        return_logits: bool,
        **gen_kwargs_passed # Pass through generate kwargs like stopping_criteria
    ) -> List[Dict[str, Any]]:
        
        results = []
        
        # Handle vision inputs if this is a vision model
        if self.is_vision_model and len(batch_items_original) == 1:
            # For vision models, process one item at a time for now
            item = batch_items_original[0]
            text_input, images = self._extract_text_and_images(item)
            
            # Use the preprocessed prompt if available, otherwise use extracted text
            if prompts:
                text_input = prompts[0]
            
            encoded = self._prepare_vision_inputs(text_input, images)
            input_lens = [encoded["attention_mask"].sum(dim=1).item()] if "attention_mask" in encoded else [len(encoded.get("input_ids", []))]
        else:
            # Standard text-only processing
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True, # Ensure truncation if prompts are too long
                max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else 2048 # Default max length
            )
            input_lens = encoded["attention_mask"].sum(dim=1).tolist()
        
        # Move to device if not distributed. If distributed, inputs stay on CPU.
        # The model.generate() call handles internal transfers.
        if not self.is_model_distributed and self.device and self.device != "cpu":
            encoded_on_device = {k: v.to(self.device) for k, v in encoded.items()}
        else: # Distributed model or CPU model, keep inputs on CPU
            encoded_on_device = encoded

        current_gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        }
        current_gen_kwargs.update(gen_kwargs_passed) # Add/override with passed kwargs

        if return_logits: # This is the return_logits from generate_batch
            current_gen_kwargs.update({
                "return_dict_in_generate": True,
                "output_scores": True,
            })

        with torch.no_grad():
            outputs = self.model.generate(**encoded_on_device, **current_gen_kwargs)
        
        if isinstance(outputs, dict): # If return_dict_in_generate was True
            sequences = outputs.sequences
            scores_list = outputs.scores # Tuple of tensors, one per generation step
        else: # sequences is a tensor
            sequences = outputs
            scores_list = None

        batch_size_actual = sequences.shape[0]
        for i in range(batch_size_actual):
            item = batch_items_original[i] # Get the original item
            input_len = input_lens[i]
            
            # Generated token IDs (excluding prompt)
            # Sequences might be on GPU, move to CPU for decoding and further processing if necessary
            gen_ids = sequences[i, input_len:].cpu() 
            
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            final_answer = generated_text
            chain_of_thought = None

            if self.cot and self.cot_parser is not None:
                # The CoT parser expects the generated text part.
                # If cot_trigger was added to prompt, generated_text is the part after it.
                try:
                    chain_of_thought, final_answer = self.cot_parser(generated_text)
                except Exception as e_parse:
                    logger.debug(f"CoT parser failed for generated text '{generated_text[:100]}...': {e_parse}. Using full text as answer.")
                    final_answer = generated_text # Fallback

            item["prediction"] = final_answer
            if chain_of_thought is not None:
                item["chain_of_thought"] = chain_of_thought
            
            if return_logits and scores_list is not None:
                # scores_list contains logits for each token *at each generation step* for the whole batch
                # Each element in scores_list is a tensor of shape (batch_size, vocab_size)
                log_probs_per_token = []
                sum_log_prob = 0.0
                
                # gen_ids are already on CPU.
                # scores_list elements are likely on the model's device(s).
                
                for t, step_scores_batch in enumerate(scores_list): # step_scores_batch is (batch_size, vocab_size)
                    if t >= len(gen_ids): # Should not exceed length of generated tokens
                        break
                    
                    # Get scores for the current item in the batch, move to CPU if not already.
                    step_score_item = step_scores_batch[i].cpu() # Shape: (vocab_size)
                    step_log_probs = F.log_softmax(step_score_item, dim=-1)
                    
                    token_id_at_step = gen_ids[t].item() # Current generated token ID for this item
                    token_log_prob = step_log_probs[token_id_at_step].item()
                    
                    sum_log_prob += token_log_prob
                    log_probs_per_token.append(token_log_prob)
                
                item["logits"] = {
                    "sum_log_prob": sum_log_prob,
                    "token_log_probs": log_probs_per_token,
                    "tokens": self.tokenizer.convert_ids_to_tokens(gen_ids), # gen_ids is already a list/tensor of IDs
                }
            results.append(item)
        return results