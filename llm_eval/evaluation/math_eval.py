import re
from typing import List, Dict, Any, Optional, Union
from .base import BaseEvaluator
from . import register_evaluator
from math_verify import parse, verify
from math_verify import LatexExtractionConfig, ExprExtractionConfig
from llm_eval.utils.logging import get_logger
from llm_eval.utils.prompt_template import extract_final_answer  # Import extract_final_answer
import logging
from tqdm import tqdm

logger = get_logger(name="math_match", level=logging.INFO)

@register_evaluator("math_match")
class MathMatchEvaluator(BaseEvaluator):
    """
    Mathematical expression evaluator that uses math_verify to compare predictions 
    against reference answers. It handles LaTeX and mathematical expressions, 
    comparing them for mathematical equivalence rather than simple string equality.
    
    Features:
    - Supports both LaTeX and plain mathematical expressions.
    - Handles set operations, equations, and other mathematical notations.
    - Can extract final answers from Chain-of-Thought (CoT) responses using extract_final_answer.
    - Stores extracted and parsed answers for debugging or analysis.
    
    Example Usage:
        from llm_eval.evaluation.math_eval import MathMatchEvaluator
        
        evaluator = MathMatchEvaluator(
            latex_only=True,           # If you're only dealing with LaTeX
            extract_final_answer=True  # To extract from Chain-of-Thought responses
        )
        
        # Example evaluation
        results = evaluator.evaluate_predictions([
            {
                "prediction": "Answer: \\boxed{1,2,3,4\\text{inch}}",
                "reference": "${1,3} \\cup {2,4}$"
            }
        ])
        
        # results will contain:
        # {
        #     "accuracy": 1.0,           # if expressions are mathematically equivalent
        #     "parse_failure_rate": 0.0,
        #     "verify_failure_rate": 0.0
        # }
    """
    name = "math_match"

    def __init__(
        self,
        latex_only: bool = True,  # If True, only use LaTeX extraction
        expr_only: bool = False,  # If True, only use expression extraction
        extract_final_answer: bool = True,  # Extract final answer using extract_final_answer
        answer_patterns: List[str] = None,  # Custom regex patterns for answer extraction (not used in this implementation)
        *args,
        **kwargs
    ):
        """
        Args:
            latex_only (bool):
                If True, only attempts LaTeX extraction. (Default: True)
            expr_only (bool):
                If True, only attempts expression extraction. (Default: False)
            extract_final_answer (bool):
                If True, extracts the final answer using keywords like "Answer:" or "정답:".
            answer_patterns (List[str] | None):
                Custom regex patterns for answer extraction (not used in this implementation).
        """
        super().__init__(*args, **kwargs)
        self.latex_only = latex_only
        self.expr_only = expr_only
        self.extract_final_answer = extract_final_answer
        # Although answer_patterns are defined, they are not used directly here.
        self.answer_patterns = answer_patterns or [
            r"정답\s*:\s*(.*?)(?:\n|$)",
            r"Answer\s*:\s*(.*?)(?:\n|$)"
        ]

        # Configure extraction based on settings
        self.extraction_config = []
        if latex_only:
            self.extraction_config.append(LatexExtractionConfig())
        elif expr_only:
            self.extraction_config.append(ExprExtractionConfig())
        else:
            self.extraction_config.extend([
                LatexExtractionConfig(),
                ExprExtractionConfig()
            ])

    def extract_answer(self, text: str) -> str:
        """
        Extracts the final answer from a Chain-of-Thought (CoT) text.
        If extract_final_answer is enabled, it uses the extract_final_answer function
        to retrieve the answer following keywords like "정답:" or "Answer:".
        If the extracted answer spans multiple lines, only the first non-empty line is returned.
        """
        if not text or not self.extract_final_answer:
            return text

        # Use extract_final_answer function to get the final answer portion
        extracted = extract_final_answer(text)
        if extracted == text:
            # No answer marker: keep the whole output so math_verify can
            # find the answer (e.g. a trailing \boxed{}) anywhere in it.
            return text
        # Split into lines and return the first non-empty line
        lines = [line.strip() for line in extracted.splitlines() if line.strip()]
        return lines[0] if lines else extracted

    def parse_math(self, text: str, is_reference: bool = False) -> Any:
        """
        Parses mathematical expressions using math_verify.
        Returns the parsed mathematical object or None if parsing fails.

        References are usually bare expressions without LaTeX delimiters
        (e.g. "18.0"), which LaTeX extraction does not pick up, so a
        reference that yields nothing is retried wrapped in "$...$".
        """
        try:
            parsed = parse(text, extraction_config=self.extraction_config)
            if not parsed and is_reference and text.strip():
                parsed = parse(f"${text.strip()}$", extraction_config=self.extraction_config)
            # math_verify returns an empty list when it finds nothing.
            return parsed if parsed else None
        except Exception as e:
            logger.warning(f"Failed to parse mathematical expression: {text}")
            logger.warning(f"Error: {str(e)}")
            return None

    def verify_equivalent(self, pred: Any, ref: Any) -> bool:
        """
        Verifies if two mathematical expressions are equivalent.
        Returns False if verification fails.
        """
        if pred is None or ref is None:
            return False

        try:
            return verify(ref, pred)
        except Exception as e:
            logger.warning(f"Verification failed between: {pred} and {ref}")
            logger.warning(f"Error: {str(e)}")
            return False

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluates mathematical expressions for equivalence and calculates accuracy.
        Additionally, stores extracted and parsed answers for debugging or analysis.
        
        For each sample, an "evaluation" field is added with:
            - "extracted_pred": The final extracted prediction text.
            - "extracted_ref": The final extracted reference text.
            - "parsed_pred": The parsed prediction object.
            - "parsed_ref": The parsed reference object.
            - "is_correct": Boolean indicating whether the prediction is mathematically equivalent to the reference.
            - "fallback_used": Boolean indicating if fallback string comparison was used.
        
        Returns:
            A dictionary containing metrics:
                - "accuracy": Correct predictions / total samples.
                - "parse_failure_rate": Fraction of samples where parsing failed.
                - "verify_failure_rate": Fraction of samples where verification failed.
        """
        total = len(samples)
        correct = 0
        parse_failures = 0
        verify_failures = 0

        for sample in tqdm(samples, desc="Math-Match Evaluation"):
            # Extract final answers
            extracted_pred = self.extract_answer(str(sample.get("prediction", "")))
            extracted_ref = self.extract_answer(str(sample.get("reference", "")))

            # Parse mathematical expressions
            parsed_pred = self.parse_math(extracted_pred)
            parsed_ref = self.parse_math(extracted_ref, is_reference=True)

            fallback_used = False
            # If parsing fails for either expression, fall back to string equality
            if parsed_pred is None or parsed_ref is None:
                logger.warning(f"Parse failure for sample:\npred: {extracted_pred}\nref: {extracted_ref}\nFalling back to string match.")
                is_correct = extracted_pred.strip() == extracted_ref.strip()
                fallback_used = True
                parse_failures += 1
            else:
                try:
                    is_correct = self.verify_equivalent(parsed_pred, parsed_ref)
                except Exception as e:
                    logger.warning(f"Verification exception for sample:\npred: {parsed_pred}\nref: {parsed_ref}\nFalling back to string match.")
                    is_correct = extracted_pred.strip() == extracted_ref.strip()
                    fallback_used = True
                    verify_failures += 1

            if is_correct:
                correct += 1

            sample["evaluation"] = {
                "extracted_pred": extracted_pred,
                "extracted_ref": extracted_ref,
                # Stored as strings: sympy objects are not JSON serializable.
                "parsed_pred": [str(x) for x in parsed_pred] if parsed_pred else None,
                "parsed_ref": [str(x) for x in parsed_ref] if parsed_ref else None,
                "is_correct": is_correct,
                "fallback_used": fallback_used
            }

        metrics = {
            "accuracy": correct / total if total else 0.0,
            "parse_failure_rate": parse_failures / total if total else 0.0,
            "verify_failure_rate": verify_failures / total if total else 0.0
        }

        return metrics
