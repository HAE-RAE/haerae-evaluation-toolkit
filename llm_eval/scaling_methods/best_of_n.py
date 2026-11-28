from typing import List, Dict, Any
from llm_eval.models.multi import MultiModel
from .base import BaseScalingMethod
from . import register_scaling_method
from llm_eval.utils.logging import get_logger
from tqdm import tqdm
import logging
import math

logger = get_logger(name="best_of_n", level=logging.INFO)

@register_scaling_method("best_of_n")
class BestOfN(BaseScalingMethod):
    """
    A scaling method that samples N times for each sample and selects the best answer
    according to a reward or log-prob score.
    """

    def __init__(self, model: MultiModel = None, n: int = 5, batch_size: int = 1, **kwargs):
        """
        Args:
            model (MultiModel): The model backend (could be MultiModel).
            n (int): Number of sampling iterations per sample.
            batch_size (int): Number of samples to process in one batch.
            kwargs: Additional parameters that may be passed from BaseScalingMethod.
        """
        super().__init__(model=model, **kwargs)
        if n < 1 or batch_size < 1:
            raise ValueError("n and batch_size must be positive")
        self.n = n
        self.batch_size = batch_size

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for Best-of-N scaling.

        Args:
            data: A list of samples, each a dict with at least {"input":..., "reference":...}.

        Returns:
            A list of the same structure with "prediction" updated to the selected best candidate
            for each sample.
        """
        if self.model is None:
            raise ValueError("BestOfN scaling requires a 'model' instance.")
        
        score_batch = getattr(self.model, "score_batch", None)
        if isinstance(self.model, MultiModel) and self.model.reward_model is None:
            score_batch = None

        for start in tqdm(range(0, len(data), self.batch_size), desc="Best-of-N"):
            batch = data[start:start + self.batch_size]
            best_scores = [float("-inf")] * len(batch)
            best_predictions = [""] * len(batch)
            for round_index in range(self.n):
                outputs = self.model.generate_batch(
                    [sample.copy() for sample in batch],
                    return_logits=score_batch is None,
                    batch_size=self.batch_size,
                    show_progress=False,
                )
                if len(outputs) != len(batch):
                    raise ValueError("Best-of-N requires one generated candidate per input")
                scored = score_batch(outputs) if score_batch is not None else outputs
                for i, (result, scores) in enumerate(zip(outputs, scored, strict=True)):
                    if score_batch is None:
                        score = (scores.get("logits") or {}).get("sum_log_prob")
                    else:
                        score = scores.get("reward")
                    if score is None or math.isnan(float(score)):
                        field = "reward" if score_batch is not None else "logits.sum_log_prob"
                        raise ValueError(f"Best-of-N requires a numeric {field} for each candidate")
                    score = float(score)
                    if round_index == 0 or score > best_scores[i]:
                        best_scores[i] = score
                        best_predictions[i] = result["prediction"]

            for sample, prediction in zip(batch, best_predictions):
                sample["prediction"] = prediction
        return data
