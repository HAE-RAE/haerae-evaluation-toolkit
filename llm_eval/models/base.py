from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from llm_eval.utils.prompt_template import default_cot_parser


class BaseModel:
    """
    Abstract base class that all model backends must inherit.

    Required method to implement:
      - generate_batch(self, inputs, return_logits=False) -> List[Dict[str, Any]]
        * inputs: [{"input": str, "reference": str, ...}, ...]

        * Returns: [{"input":..., "reference":...,
                      "prediction":...,        # Final string output from the model
                      "logits": (optional)..., # if return_logits=True
                      ...}, ...]
    """

    def __init__(
        self,
        cot_trigger: Optional[str] = "Let's think step by step.",
        cot_parser: Optional[Callable[[str],
                                      Tuple[str, str]]] = default_cot_parser,
        **kwargs
    ):
        # Parameters received when calling super().__init__(...) in a subclass
        self.cot_trigger = cot_trigger
        self.cot_parser = cot_parser

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        cot: bool = False,
        batch_size: Optional[Union[int, str]] = "auto",
        until: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Method to generate text (answers) from the LLM.
        Args:
            inputs: [{"input": str, "reference": str, ...}, ...]
            return_logits: If True, additional information such as logits or logprobs may be returned.

                 2) the model may include its reasoning in the "chain_of_thought" field.
        Returns:
            The same list (or a copy) with each element augmented as follows:
            [
              {
                "input": ...,
                "reference": ...,
                "prediction": <generated answer>,

                "chain_of_thought": "...(intermediate reasoning)..."
                ...
              },
              ...
            ]
        """
        raise NotImplementedError(
            "Subclasses must implement generate_batch().")


class BaseJudge:
    """
    Abstract base class for the Judge model (LLM-as-a-Judge).
    It takes generated text (answers) as input and evaluates their quality/appropriateness.
    For example, it can be used for chain-of-thought based self-consistency evaluation, star ratings (1-5 points), etc.
    """

    def __init__(self, **kwargs):
        pass

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Args:
            inputs: [{"input": ..., "prediction": ..., "reference": ...}, ...]
            - Typically, the 'prediction' (generated answer) is used for quality evaluation.
            Returns:
            [{"judge_score": float or int, "judge_explanation": str, ...}, ...]
            - Returns each sample with an added evaluation score/assessment.

        """
        raise NotImplementedError("Subclasses must implement judge_batch().")


class BaseRewardModel:
    """
    Abstract class dedicated to Reward models (usable in DVTS, etc.).
    It estimates a scalar reward value from a text answer.
    """

    def __init__(self, **kwargs):
        pass

    def score_batch(
        self,
        inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Args:
            inputs: [{"input":..., "prediction":..., "reference":...}, ...]
                    - Typically, the 'prediction' is used as input to compute the reward score.
        Returns:
            [{"reward": float, ...}, ...]
            - Each sample is augmented with a 'reward' field.
        """
        raise NotImplementedError("Subclasses must implement score_batch().")
