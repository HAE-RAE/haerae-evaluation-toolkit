import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from .base import BaseModel
from . import register_model
from llm_eval.utils.logging import get_logger
from llm_eval.utils.prompt_template import extract_final_answer

logger = get_logger(name="huggingface", level=logging.INFO)

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_sequences: List[str], tokenizer):
        super().__init__()
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated = self.tokenizer.decode(input_ids[0])
        return any(stop_seq in generated for stop_seq in self.stop_sequences)

@register_model("huggingface")
class HuggingFaceModel(BaseModel):
    """
    A backend model class that uses HuggingFace Transformers.

    Main points:
      - If `return_logits=True`, we call `model.generate(..., return_dict_in_generate=True, output_scores=True)`
        so that "scores" (logits per step) are included in the output, allowing us to compute log probabilities.
      - If a `cot_parser` function is provided, we can separate chain-of-thought text from the final answer.
      - This code avoids using the older GenerationOutput class, which may not be available in recent Transformers versions.
    
    Args:
        model_name_or_path (str): HF Hub model ID or local path.
        device (str): 'cpu', 'cuda', 'cuda:0', etc.
        max_new_tokens (int): Maximum new tokens to generate in one call.
        cot_trigger (str|None): Optional CoT (Chain-of-Thought) trigger appended to the prompt. If None, no CoT trigger.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        do_sample (bool): If True, sampling mode; if False, greedy generation.
        cot_parser (callable|None): A function that takes a string (generated text) and returns (chain_of_thought, final_answer).
        **kwargs: Additional parameters as needed.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
        cot_trigger: Optional[str] = "Let's think step by step.",
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        cot_parser: Optional[Callable[[str], Tuple[str, str]]] = extract_final_answer,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"[HuggingFaceModel] Loading tokenizer/model from {model_name_or_path}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.eval()

        # Device setup
        self.device = device
        if self.device != "cpu":
            self.model.to(self.device)
            logger.info(f"[HuggingFaceModel] Model moved to {self.device}")

        # Inference hyperparameters
        self.max_new_tokens = max_new_tokens
        self.cot_trigger = cot_trigger
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.cot_parser = cot_parser

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        cot: bool = False,
        batch_size: Optional[Union[int, str]] = 1 # auto
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of inputs to generate text outputs and updates each item with the final prediction.

        Args:
            inputs (List[Dict[str, Any]]): 
                A list of items, each containing at least {"input": str, "reference": str, ...}.
            return_logits (bool): 
                If True, compute log probabilities for the generated tokens and store them in item["logits"].
            cot (bool): 
                If True, append self.cot_trigger to the original prompt (if defined) and parse out chain-of-thought.
            batch_size (int | str): 
                The batch size to use. If "auto", it starts with a default and reduces if OOM occurs.

        Returns:
            List[Dict[str, Any]]:
                An updated list where each item has new fields:
                  - "prediction": the generated final answer
                  - "chain_of_thought": optional CoT text if parsed
                  - "logits": optional dict containing log probabilities if `return_logits=True`.
        """
        # Determine initial batch size
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            auto_mode = True
            current_bs = 128  # "auto" initial batch size
            logger.info(f"[HuggingFaceModel] Batch size set to 'auto'. Starting with batch size {current_bs}.")
        else:
            auto_mode = False
            current_bs = batch_size if batch_size is not None else len(inputs)
            logger.info(f"[HuggingFaceModel] Batch size set to {current_bs}.")

        while True:
            try:
                results = []
                # Process in chunks
                for start in range(0, len(inputs), current_bs):
                    batch_items = inputs[start:start + current_bs]

                    # Build prompts
                    prompts = [
                        f"{item['input']}\n{self.cot_trigger}\n" if (cot and self.cot_trigger) else item["input"]
                        for item in batch_items
                    ]

                    # Tokenize (with padding/truncation)
                    encoded = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    # Compute input lengths
                    input_lens = encoded["attention_mask"].sum(dim=1).tolist()

                    # Move to device
                    if self.device != "cpu":
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}

                    # Generation arguments
                    gen_kwargs = {
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "do_sample": self.do_sample,
                    }
                    if return_logits:
                        gen_kwargs.update({
                            "return_dict_in_generate": True,
                            "output_scores": True,
                        })

                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(**encoded, **gen_kwargs)

                    # For the latest versions of HF, 'outputs' can be either:
                    # - A dict with "sequences" and "scores" if return_dict_in_generate=True
                    # - A standard tensor if return_dict_in_generate=False
                    if return_logits and isinstance(outputs, dict):
                        sequences = outputs["sequences"]
                        scores_list = outputs["scores"]  # a list of (batch_size, vocab_size) tensors
                    elif return_logits and not isinstance(outputs, dict):
                        sequences = outputs
                        scores_list = None
                        logger.warning("[HuggingFaceModel] `return_dict_in_generate=True` was set, but output is not a dict. No scores available.")
                    else:
                        sequences = outputs
                        scores_list = None

                    # Process each sample in the batch
                    batch_size_actual = sequences.shape[0]
                    for i in range(batch_size_actual):
                        item = batch_items[i]
                        input_len = input_lens[i]
                        gen_ids = sequences[i][input_len:]  # Extract generated portion
                        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                        # Parse CoT if needed
                        final_answer = generated_text
                        chain_of_thought = None
                        if cot and self.cot_parser is not None:
                            chain_of_thought, final_answer = self.cot_parser(generated_text)

                        item["prediction"] = final_answer
                        if chain_of_thought is not None:
                            item["chain_of_thought"] = chain_of_thought

                        # Compute log probabilities if 'scores_list' is available
                        if return_logits and scores_list is not None:
                            log_probs_per_token = []
                            sum_log_prob = 0.0
                            token_ids = gen_ids.tolist()

                            # scores_list[t] => shape (batch_size, vocab_size)
                            # Each step: compute log prob for the token_id
                            for t, step_scores in enumerate(scores_list):
                                if t >= len(token_ids):
                                    break
                                step_score = step_scores[i]  # shape (vocab_size,)
                                step_log_probs = F.log_softmax(step_score, dim=-1)
                                token_log_prob = step_log_probs[token_ids[t]].item()
                                sum_log_prob += token_log_prob
                                log_probs_per_token.append(token_log_prob)

                            item["logits"] = {
                                "sum_log_prob": sum_log_prob,
                                "token_log_probs": log_probs_per_token,
                                "tokens": self.tokenizer.convert_ids_to_tokens(token_ids),
                            }

                        results.append(item)

                return results

            except RuntimeError as e:
                # Handle out-of-memory by reducing batch size if in auto mode
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if auto_mode:
                        if current_bs > 1:
                            current_bs = max(1, current_bs // 2)
                            logger.warning(f"OOM detected: reducing batch size to {current_bs}.")
                        else:
                            logger.error("Batch size is 1 but OOM still occurs.")
                            raise RuntimeError("Out of memory even with batch size = 1.") from e
                    else:
                        logger.error("Out of memory with the specified batch size.")
                        raise RuntimeError("Out of memory with the specified batch size.") from e
                else:
                    logger.error("A RuntimeError occurred:", exc_info=True)
                    raise

    def generate_until(
        self,
        prompt: str,
        stop_sequences: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """HuggingFace 모델을 사용하여 특정 시퀀스가 나올 때까지 텍스트 생성
        
        Args:
            prompt (str): 입력 프롬프트 텍스트
            stop_sequences (List[str]): 생성을 중단할 시퀀스 목록
            max_tokens (Optional[int]): 생성할 최대 토큰 수.
                None인 경우 self.max_new_tokens 사용
            **kwargs: 추가적인 모델 생성 파라미터

        Returns:
            str: 프롬프트를 제외한 생성된 텍스트
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        max_new_tokens = max_tokens or self.max_new_tokens
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([
                KeywordsStoppingCriteria(stop_sequences, self.tokenizer)
            ]),
            **kwargs
        )
        
        # Decode and remove prompt
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]
