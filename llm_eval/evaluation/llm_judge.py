from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging
import time
from enum import Enum
from ..models.multi import MultiModel
from .base import BaseEvaluator
from ..utils.prompt_template import JUDGE_PROMPTS


class JudgeType(Enum):
    RUBRIC_AND_RESPONSE = "rubric_and_response"
    RUBRIC_RESPONSE_AND_GOLD = "rubric_response_and_gold"
    RESPONSE_COMPARISON = "response_comparison"


@dataclass
class JudgeInput:
    judge_type: JudgeType
    model_response: str
    rubric: Optional[str] = None
    gold_response: Optional[str] = None
    model_response_b: Optional[str] = None


class ResponseParser:
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        """Base parser interface"""
        raise NotImplementedError


class RubricScoreParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        import re
        score_pattern = r"\[\[score:\s*(\d+\.?\d*)\]\]"
        match = re.search(score_pattern, response)
        if not match:
            raise ValueError(f"No valid score found in response: {response}")
        
        return {
            "score": float(match.group(1)),
            "model_name": model_name if model_name else "unknown"
        }


class PairwiseComparisonParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        result = {}
        if "[[A]]" in response:
            result = {"winner": "A"}
        elif "[[B]]" in response:
            result = {"winner": "B"}
        elif "[[C]]" in response:
            result = {"winner": "tie"}
        else:
            raise ValueError(f"No valid verdict found in response: {response}")
        
        result["model_name"] = model_name if model_name else "unknown"
        return result


class GoldComparisonParser(ResponseParser):
    def parse(self, response: str, model_name: str = None) -> Dict[str, Any]:
        if not response:
            raise ValueError("Response is None")
        if "[[true]]" in response.lower():
            return {
                "correct": True,
                "step": -1,
                "model_name": model_name if model_name else "unknown"
            }
        elif "[[false]]" in response.lower():
            import re
            step_pattern = r"step:\s*\[(\d+)\]"
            match = re.search(step_pattern, response)
            if match:
                return {
                    "correct": False,
                    "step": int(match.group(1)),
                    "model_name": model_name if model_name else "unknown"
                }
        raise ValueError(f"No valid verdict found in response: {response}")


class MultiLLMJudge:
    def __init__(
        self,
        models_config: List[Dict[str, Any]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        aggregation_strategy: str = "majority",
        logger: Optional[logging.Logger] = None
    ):
        self.multi_model = MultiModel(items_config=models_config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.aggregation_strategy = aggregation_strategy
        self.logger = logger or logging.getLogger(__name__)
        
        self.parsers = {
            JudgeType.RUBRIC_AND_RESPONSE: RubricScoreParser(),
            JudgeType.RUBRIC_RESPONSE_AND_GOLD: GoldComparisonParser(),
            JudgeType.RESPONSE_COMPARISON: PairwiseComparisonParser()
        }
        
        self.prompt_templates = JUDGE_PROMPTS

    def _format_prompt(self, judge_input: JudgeInput) -> str:
        if judge_input.judge_type == JudgeType.RUBRIC_AND_RESPONSE:
            return f"""You are an expert evaluator. Your task is to evaluate the given response based on the rubric and provide a score.

IMPORTANT: You must format your response exactly like this example:
Based on the rubric, this response deserves [[score: 7]].

Rubric:
{judge_input.rubric}

Response to evaluate:
{judge_input.model_response}

Provide your evaluation with the score in the specified format:"""
            
        elif judge_input.judge_type == JudgeType.RESPONSE_COMPARISON:
            return f"""You are an expert evaluator. Your task is to compare two responses and choose the better one.

IMPORTANT: You must format your verdict exactly like this:
- Use [[A]] to choose the first response
- Use [[B]] to choose the second response
- Use [[C]] if they are equally good

Response A:
{judge_input.model_response}

Response B:
{judge_input.model_response_b}

Provide your verdict in the specified format:"""
            
        elif judge_input.judge_type == JudgeType.RUBRIC_RESPONSE_AND_GOLD:
            return f"""You are an expert evaluator. Please evaluate if the following response matches the gold standard answer.
Compare step by step and provide your verdict as [[true]] if correct or [[false]] step: [X] if incorrect.

Rubric:
{judge_input.rubric}

Gold Response:
{judge_input.gold_response}

Model Response to evaluate:
{judge_input.model_response}

Provide your evaluation in the specified format:"""
        
        raise ValueError(f"Unsupported judge type: {judge_input.judge_type}")

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            raise ValueError("No results to aggregate")

        if self.aggregation_strategy == "majority":
            if "score" in results[0]:  # Rubric scoring
                scores = [result.get("score", 0) for result in results if "score" in result]
                models = [result.get("model_name", "unknown") for result in results if "score" in result]
                if not scores:
                    raise ValueError("No valid scores found in results")
                average_score = sum(scores) / len(scores)
                return {
                    "score": average_score,
                    "individual_scores": scores,
                    "model_names": models,
                    "num_models": len(scores)
                }
            elif "winner" in results[0]:  # Pairwise comparison
                winners = [result.get("winner") for result in results if "winner" in result]
                models = [result.get("model_name", "unknown") for result in results if "winner" in result]
                from collections import Counter
                winner_counts = Counter(winners)
                majority_winner = max(winner_counts.items(), key=lambda x: x[1])[0]
                return {
                    "winner": majority_winner,
                    "model_names": models
                }
            elif "correct" in results[0]:  # Gold comparison
                corrects = [result.get("correct", False) for result in results if "correct" in result]
                models = [result.get("model_name", "unknown") for result in results if "correct" in result]
                majority_correct = sum(corrects) > len(corrects) / 2
                return {
                    "correct": majority_correct,
                    "model_names": models
                }
        
        elif self.aggregation_strategy == "first":
            return results[0]
        
        elif self.aggregation_strategy == "all":
            return {"results": results}
        
        raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")

    def judge(self, judge_input: JudgeInput) -> Dict[str, Any]:
        prompt = self._format_prompt(judge_input)
        parser = self.parsers[judge_input.judge_type]
        
        batch_input = [{
            "input": prompt,
            "judge_type": judge_input.judge_type.value
        }]
        
        all_results = []
        for attempt in range(self.max_retries):
            try:
                responses = self.multi_model.generate_batch(batch_input)
                self.logger.debug(f"Raw responses from models: {responses}")
                
                for response in responses:
                    self.logger.debug(f"Processing response: {response}")
                    
                    if "prediction" in response:
                        try:
                            result = parser.parse(
                                response["prediction"],
                                model_name=response.get("model_name", "unknown")
                            )
                            all_results.append(result)
                        except ValueError as e:
                            self.logger.warning(f"Failed to parse direct prediction: {str(e)}")
                    
                    if "multi_outputs" in response:
                        for model_output in response["multi_outputs"]:
                            try:
                                result = parser.parse(
                                    model_output["prediction"],
                                    model_name=model_output.get("model_name", "unknown")
                                )
                                all_results.append(result)
                            except ValueError as e:
                                self.logger.warning(f"Failed to parse response from {model_output.get('model_name', 'unknown')}: {str(e)}")
                
                if all_results:
                    return self._aggregate_results(all_results)
                
                self.logger.warning("No valid results were parsed from model responses")
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise ValueError(f"Failed to get valid judgments after {self.max_retries} attempts")
        
        raise ValueError("No valid results obtained from any model")