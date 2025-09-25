from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from llm_eval.utils.prompt_template import default_cot_parser


class BaseModel:
    """
    Abstract base class that all model backends must inherit.

    Required method to implement:
      - generate_batch(self, inputs, return_logits=False) -> List[Dict[str, Any]]
        * inputs: [{"input": str or dict, "reference": str, ...}, ...]
          - For text-only: {"input": "text prompt", "reference": "answer", ...}
          - For VQA: {"input": {"text": "question", "images": [...]}, "reference": "answer", ...}
          - For VQA (alternative): {"input": "question", "images": [...], "reference": "answer", ...}

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
            inputs: [{"input": str or dict, "reference": str, ...}, ...]
                   For VQA, input can be:
                   - {"input": {"text": "question", "images": [...]}, ...}
                   - {"input": "question", "images": [...], ...}
            return_logits: If True, additional information such as logits or logprobs may be returned.
            cot: If True, the model may include its reasoning in the "chain_of_thought" field.
        Returns:
            The same list (or a copy) with each element augmented as follows:
            [
              {
                "input": ...,
                "reference": ...,
                "prediction": <generated answer>,
                "chain_of_thought": "...(intermediate reasoning)..." (optional)
                ...
              },
              ...
            ]
        """
        raise NotImplementedError(
            "Subclasses must implement generate_batch().")
    
    def _is_vision_input(self, item: Dict[str, Any]) -> bool:
        """
        Helper method to determine if an input item contains vision data.
        
        Args:
            item: Input item dictionary
            
        Returns:
            bool: True if the item contains images, False otherwise
        """
        # Check if images are provided in the item directly
        if "images" in item and item["images"]:
            return True
        
        # Check if input is a dict with images
        if isinstance(item.get("input"), dict) and "images" in item["input"] and item["input"]["images"]:
            return True
            
        return False
    
    def _extract_text_and_images(self, item: Dict[str, Any]) -> Tuple[str, Optional[List]]:
        """
        Helper method to extract text and images from an input item.
        
        Args:
            item: Input item dictionary
            
        Returns:
            Tuple[str, Optional[List]]: (text_prompt, images_list)
        """
        # Case 1: images are in the item directly
        if "images" in item:
            text = item.get("input", "")
            if isinstance(text, dict):
                text = text.get("text", "")
            return text, item["images"]
        
        # Case 2: input is a dict with text and images
        if isinstance(item.get("input"), dict):
            input_dict = item["input"]
            text = input_dict.get("text", "")
            images = input_dict.get("images", None)
            return text, images
        
        # Case 3: text-only input
        return str(item.get("input", "")), None


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
