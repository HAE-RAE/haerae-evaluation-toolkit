import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from tqdm import tqdm

from llm_eval.models.multi import MultiModel
from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.prompt_template import JUDGE_PROMPTS, JudgeType
from llm_eval.utils.logging import get_logger

logger = get_logger(name="llm_judge", level=logging.INFO)


@dataclass
class JudgeInput:
    """
    Data structure representing the input necessary for the judge system.
    """
    input: str
    judge_type: JudgeType
    model_response: str
    rubric: Optional[str] = None
    gold_response: Optional[str] = None
    model_response_b: Optional[str] = None


class ResponseParser:
    """Base class for parsing the LLM's returned response."""
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        raise NotImplementedError


class RubricScoreParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        score_pattern = r"\[\[score:\s*(\d+\.?\d*)\]\]"
        match = re.search(score_pattern, response)
        if not match:
            raise ValueError(f"No valid score found in response: {response}")
        return {
            "score": float(match.group(1)),
            "model_name": model_name or "unknown"
        }


class PairwiseComparisonParser(ResponseParser):
    def parse(self, response: str, model_a: str = None, model_b: str = None) -> Dict[str, Any]:
        winner_pattern = r"\[\[([AB])\]\]"
        match = re.search(winner_pattern, response)
        if not match:
            raise ValueError(f"No valid winner found in response: {response}")
        winner = match.group(1)
        model_name = model_a if winner == "A" else model_b
        return {
            "winner": winner,
            "model_name": model_name or "unknown"
        }


class K2EvalResponseParser(ResponseParser):
    """
    Extracts a numeric score from a response in K2-Eval format.
    Expected format: a line like "Score: 3.5" somewhere in the text.
    """
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        pattern = r"Score:\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, response)
        if not match:
            raise ValueError(f"No valid score found in K2-Eval response: {response}")
        score = float(match.group(1))
        return {
            "score": score,
            "model_name": model_name or "unknown"
        }


class GoldComparisonParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        verdicts = set(re.findall(r"\[\[(true|false)\]\]", response.lower()))
        if len(verdicts) != 1:
            raise ValueError("Expected an unambiguous [[true]] or [[false]] verdict")
        correct = verdicts == {"true"}
        parsed = {"correct": correct, "model_name": model_name or "unknown"}
        if not correct:
            match = re.search(r"step:\s*\[(\d+)\]", response, flags=re.IGNORECASE)
            parsed["step"] = int(match.group(1)) if match else None
        return parsed


class ResponseComparisonParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        match = re.search(r'\[\[([AB])\]\]', response)
        if not match:
            raise ValueError(f"No valid verdict [[A]] or [[B]] found in response: {response}")
        winner = match.group(1)
        return {"winner": winner}


@register_evaluator("llm_judge")
class LLMJudgeEvaluator(BaseEvaluator):
    """
    Evaluator that uses an LLM-as-a-Judge approach to assess model responses.
    Judge evaluation results (raw LLM output) are stored separately so that
    the original model generation result in sample["prediction"] is preserved.
    """
    name = "llm_judge"
    has_custom_judge = True

    def __init__(
        self,
        model: MultiModel,
        default_judge_type: Union[str, JudgeType] = "rubric_and_response",
        **kwargs
    ):
        super().__init__()
        self.default_judge_type = (
            JudgeType(default_judge_type)
            if isinstance(default_judge_type, str)
            else default_judge_type
        )
        self.multi_judge_model = model
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser(),
            JudgeType.K2_EVAL: K2EvalResponseParser()
        }
        self.prompt_templates = JUDGE_PROMPTS

    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        judge_type = sample.get("judge_type", self.default_judge_type.value)
        template = self.prompt_templates.get(JudgeType(judge_type))
        if template is None:
            raise ValueError(f"No template found for judge_type: {judge_type}")
        return template.format(**sample)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        if not samples:
            return {}

        total_score = 0.0
        score_count = 0
        total_correct = 0
        verdict_count = 0
        parse_failures = 0

        batch_inputs = []
        batch_indices = []

        for i, sample in enumerate(samples):
            try:
                judge_type_str = sample.get("judge_type", self.default_judge_type.value)
                j_type = JudgeType(judge_type_str)

                # Prepare prompt based on judge type.
                if j_type == JudgeType.RESPONSE_COMPARISON:
                    filled_prompt = (
                        f"Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants. "
                        f"Choose the assistant that follows the instructions and answers the question better.\n\n"
                        f"### Instruction:\n{sample.get('input', '').strip()}\n\n"
                        f"### Response A:\n{sample.get('prediction', '').strip()}\n\n"
                        f"### Response B:\n{sample.get('model_response_b', '').strip()}\n\n"
                        "Provide your verdict in this exact format: [[A]] or [[B]]."
                    )
                else:
                    template = self.prompt_templates.get(j_type)
                    if template is None:
                        raise ValueError(f"No template found for judge_type: {j_type}")
                    filled_prompt = template.format(
                        input=sample.get("input", "").strip(),
                        rubric=sample.get("rubric", "").strip(),
                        response=sample.get("prediction", "").strip(),
                        gold=sample.get("reference", "").strip(),
                        response_b=sample.get("model_response_b", "").strip()
                    )

                batch_inputs.append({"input": filled_prompt})
                batch_indices.append(i)
            except Exception as e:
                logger.error(f"Error preparing prompt: {e}")

        if batch_inputs:
            try:
                logger.info(f"LLMJudgeEvaluator: Calling judge_batch for {len(batch_inputs)} samples")
                judge_responses = self.multi_judge_model.judge_batch(batch_inputs)
                # Process each judge response
                for response_idx, sample_idx in enumerate(batch_indices):
                    sample = samples[sample_idx]
                    judge_response = judge_responses[response_idx]["prediction"]
                    
                    # Instead of overwriting the original generation result,
                    # we store the judge output separately.
                    sample["judge_evaluation"] = judge_response
                    
                    j_type = JudgeType(sample.get("judge_type", self.default_judge_type.value))
                    
                    # For K2_EVAL judge type, use K2EvalResponseParser to extract score.
                    if j_type == JudgeType.K2_EVAL:
                        parser = self.parsers.get(JudgeType.K2_EVAL)
                        try:
                            parsed = parser.parse(judge_response)
                            sample["judge_score"] = parsed["score"]
                            total_score += parsed["score"]
                            score_count += 1
                        except Exception as e:
                            logger.error(f"Error parsing K2-Eval response: {e}")
                    elif j_type == JudgeType.RUBRIC_AND_RESPONSE:
                        if "judge_score" in judge_responses[response_idx]:
                            score = judge_responses[response_idx]["judge_score"]
                            if isinstance(score, (int, float)):
                                sample["judge_score"] = score
                                total_score += score
                                score_count += 1
                    elif j_type in (JudgeType.RUBRIC_RESPONSE_AND_GOLD, JudgeType.RESPONSE_COMPARISON):
                        verdict_count += 1
                        evaluation = {"raw_output": judge_response}
                        sample["evaluation"] = evaluation
                        try:
                            parsed = self.parsers[j_type].parse(judge_response or "")
                        except ValueError as exc:
                            parse_failures += 1
                            sample["judge_correct"] = None
                            sample.pop("judge_winner", None)
                            evaluation.update(is_correct=None, error=str(exc))
                            continue

                        if j_type == JudgeType.RESPONSE_COMPARISON:
                            winner = parsed["winner"]
                            sample["judge_winner"] = evaluation["winner"] = winner
                            model_key = "model_a" if winner == "A" else "model_b"
                            parsed["model_name"] = sample.get(model_key, "unknown")
                            is_correct = sample.get("reference") == winner
                        else:
                            is_correct = parsed["correct"]

                        evaluation.update(parsed=parsed, is_correct=is_correct)
                        sample["judge_correct"] = is_correct
                        total_correct += int(is_correct)

            except Exception as e:
                logger.error(f"Error in judge_batch: {e}")
                return {"error": str(e)}

        metrics = {}
        if score_count > 0:
            metrics["average_judge_score"] = total_score / score_count
        if verdict_count:
            metrics["judge_accuracy"] = total_correct / verdict_count
            metrics["judge_parse_failures"] = parse_failures

        return metrics
