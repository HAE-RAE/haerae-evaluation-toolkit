import re
import string
from typing import List, Dict, Any, Optional, Union

from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import extract_final_answer
from llm_eval.utils.logging import get_logger
from tqdm import tqdm
import logging

logger = get_logger(name="partial_match", level=logging.INFO)

@register_evaluator("partial_match")
class PartialMatchEvaluator(BaseEvaluator):
    """
    Evaluator that checks whether the normalized prediction partially matches the 
    normalized reference. In other words, the prediction is considered correct if 
    the normalized reference is found as a substring within the normalized prediction.
    
    For MCQA tasks:
      - When mcqa is enabled and an "options" field is present in the sample, the evaluator 
        first verifies that the normalized reference is one of the candidate options.
      - Only if the reference is a valid option does it then check if the normalized prediction 
        contains the normalized reference.
    
    This ensures that the evaluation does not erroneously consider a prediction correct 
    simply because it contains any candidate option, but rather that it correctly reflects 
    the ground truth answer.
    """

    name = "partial_match"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        regexes_to_ignore: Optional[List[str]] = None,
        extract_final_answer: bool = True,
        mcqa: bool = True,  # Enable specialized handling for MCQA tasks
        *args,
        **kwargs
    ):
        """
        Args:
            ignore_case (bool): If True, perform case-insensitive matching. (Default: True)
            ignore_punctuation (bool): If True, remove punctuation from texts before matching.
            ignore_numbers (bool): If True, remove numeric characters before matching.
            regexes_to_ignore (List[str] | None): List of regex patterns to remove from text.
            extract_final_answer (bool): If True, extract the text after 'Answer:' from outputs.
            mcqa (bool): If True and an "options" field exists, use MCQA-specific matching criteria.
        """
        super().__init__(*args, **kwargs)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers
        self.regexes_to_ignore = regexes_to_ignore or []
        self.extract_final_answer = extract_final_answer
        self.mcqa = mcqa

    def _normalize_text(self, text: str) -> str:
        """
        Normalize the text by performing the following operations:
          1) Remove any patterns specified in regexes_to_ignore.
          2) Convert text to lowercase if ignore_case is enabled.
          3) Remove punctuation if ignore_punctuation is enabled.
          4) Remove numbers if ignore_numbers is enabled.
          5) Normalize whitespace by stripping extra spaces.
        """
        # Remove regex-specified patterns
        for pattern in self.regexes_to_ignore:
            text = re.sub(pattern, "", text)
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            repl_table = str.maketrans("", "", string.punctuation)
            text = text.translate(repl_table)
        if self.ignore_numbers:
            repl_table = str.maketrans("", "", string.digits)
            text = text.translate(repl_table)
        # Normalize whitespace
        text = " ".join(text.strip().split())
        return text

    def parse_prediction(self, raw_output: Any) -> str:
        """
        Process the model's raw output:
          - If extract_final_answer is enabled, extract the portion after 'Answer:'.
          - If the output consists of multiple lines, only the first non-empty line is considered.
          - Normalize the resulting text.
        """
        if raw_output is None:
            raw_output = ""
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)
        if self.extract_final_answer:
            raw_output = extract_final_answer(raw_output)
        lines = raw_output.strip().splitlines()
        if lines:
            raw_output = lines[0]
        return self._normalize_text(raw_output)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes accuracy based on partial matching between the prediction and the reference.
        
        For non-MCQA tasks:
          - A prediction is correct if the normalized reference is a substring of the normalized prediction.
        
        For MCQA tasks:
          - First, the evaluator checks that the normalized reference is among the normalized candidate options.
          - Then, it considers the prediction correct only if the normalized prediction contains 
            the normalized reference.
        
        Each sample is augmented with an "evaluation" dictionary containing:
          - The normalized prediction.
          - The normalized reference.
          - A boolean indicating correctness.
          - The original options (if any).
        
        Returns:
            A dictionary with the computed accuracy, e.g. {"accuracy": 0.85}.
        """
        total = len(samples)
        correct = 0
        logger.info("Evaluating outputs using partial (containment) match with mcqa={}".format(self.mcqa))

        for sample in tqdm(samples, desc="Partial-Match Evaluation"):
            pred_norm = self.parse_prediction(sample.get("prediction", ""))
            ref_norm = self.parse_prediction(sample.get("reference", ""))

            if not ref_norm:
                raise ValueError("partial_match requires a non-empty normalized reference")

            if self.mcqa and "options" in sample and isinstance(sample["options"], list) and sample["options"]:
                # Normalize candidate options
                normalized_options = [self._normalize_text(opt) for opt in sample["options"]]
                # For MCQA tasks, the reference must be one of the candidate options and
                # the normalized prediction must contain the normalized reference.
                is_correct = (ref_norm in normalized_options) and (ref_norm in pred_norm)
            else:
                # For non-MCQA tasks, check if the normalized reference is a substring of the normalized prediction.
                is_correct = (ref_norm in pred_norm)

            if is_correct:
                correct += 1

            sample["evaluation"] = {
                "normalized_pred": pred_norm,
                "normalized_ref": ref_norm,
                "is_correct": is_correct,
                "options": sample.get("options", None)
            }

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}
