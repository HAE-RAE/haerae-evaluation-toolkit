from typing import List, Dict, Any, Optional, Union
import logging

from .base import BaseModel, BaseJudge, BaseRewardModel
from . import load_model, register_model

logger = logging.getLogger(__name__)

@register_model("multi")
class MultiModel:
    """
    A MultiModel object that can store up to one model for each of the three roles,
    and allows you to call the corresponding method as needed.

    - self.generate_model: Inherits from BaseModel (for text generation)
    - self.judge_model: Inherits from BaseJudge (LLM-as-a-Judge)
    - self.reward_model: Inherits from BaseRewardModel (for calculating reward scores)

    Example usage:
        config = {
          "generate_model": { "name": "huggingface", "params": { "model_name_or_path": "gpt2" } },
          "judge_model": { "name": "my_judge_llm", "params": {...} },
          "reward_model": None
        }
        multi_model = load_model("multi", **config)

        # Text generation
        generated = multi_model.generate_batch(data, return_logits=False)

        # Judge evaluation
        judged = multi_model.judge_batch(generated)

        # Reward scoring
        scored = multi_model.score_batch(judged)
    """

    def __init__(
        self,
        generate_model: Optional[Dict[str, Any]] = None,
        judge_model: Optional[Dict[str, Any]] = None,
        reward_model: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Args:
            generate_model: 
                - A dict of the form {"name": "huggingface", "params": {...}} to load a BaseModel implementation.
            judge_model:
                - A dict of the form {"name": "some_judge_backend", "params": {...}} to load a BaseJudge implementation.
            reward_model:
                - A dict of the form {"name": "some_reward_backend", "params": {...}} to load a BaseRewardModel implementation.
            kwargs:
                - Other arguments are ignored or can be used for future extensions.
        """
        # Instances of BaseModel, BaseJudge, BaseRewardModel or None
        self.generate_model: Optional[BaseModel] = None
        self.judge_model: Optional[BaseJudge] = None
        self.reward_model: Optional[BaseRewardModel] = None

        # Loading the generate_model
        if generate_model is not None:
            g_name = generate_model.get("name")
            g_params = generate_model.get("params", {})
            logger.info(f"[MultiModel] Loading generate model: {g_name}")
            loaded = load_model(g_name, **g_params)
            if not isinstance(loaded, BaseModel):
                raise ValueError(f"Loaded generate_model is not a BaseModel: {type(loaded)}")
            self.generate_model = loaded

        # Loading the judge_model
        if judge_model is not None:
            j_name = judge_model.get("name")
            j_params = judge_model.get("params", {})
            logger.info(f"[MultiModel] Loading judge model: {j_name}")
            loaded = load_model(j_name, **j_params)
            if not isinstance(loaded, BaseJudge):
                raise ValueError(f"Loaded judge_model is not a BaseJudge: {type(loaded)}")
            self.judge_model = loaded

        # Loading the reward_model
        if reward_model is not None:
            r_name = reward_model.get("name")
            r_params = reward_model.get("params", {})
            logger.info(f"[MultiModel] Loading reward model: {r_name}")
            loaded = load_model(r_name, **r_params)
            if not isinstance(loaded, BaseRewardModel):
                raise ValueError(f"Loaded reward_model is not a BaseRewardModel: {type(loaded)}")
            self.reward_model = loaded

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        1) If generate_model exists, perform text generation for each sample.
        2) Add a "prediction" field to each sample with the generated text.
        3) If return_logits=True, also add a "logits" field.
        4) Returns N outputs for N inputs (maintaining the same length).

        Example:
          inputs = [{"input": "Hello", "reference": "World"}, ...]
          returns -> [{"input": "Hello", "reference": "World", "prediction": "..."}, ...]
        """
        if self.generate_model is None:
            # If generate_model does not exist, return the inputs unchanged.
            return inputs

        return self.generate_model.generate_batch(
            inputs,
            return_logits=return_logits,
            **kwargs
        )

    def judge_batch(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        1) If judge_model exists, perform judge_batch processing on each sample.
        2) The judge_model (inheriting from BaseJudge) will add fields such as "judge_score" and "judge_explanation"
           to each sample.
        3) Returns N outputs for N inputs.
        """
        if self.judge_model is None:
            return inputs
        
        # If judge_model is an instance of BaseJudge, call judge_model.judge_batch(inputs)
        return self.judge_model.judge_batch(inputs, **kwargs)

    def score_batch(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        1) If reward_model exists, perform score_batch processing on each sample.
        2) The reward_model (inheriting from BaseRewardModel) will add a "reward" field to each sample.
        3) Returns N outputs for N inputs.
        """
        if self.reward_model is None:
            return inputs

        return self.reward_model.score_batch(inputs, **kwargs)
