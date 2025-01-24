from typing import List, Dict, Any
from llm_eval.models.base import BaseModel
from .base import BaseScalingMethod
from . import register_scaling_method
from transformers import pipeline
#

@register_scaling_method("best_of_n")
class BestOfN(BaseScalingMethod):
    """
    N번 샘플링하여 가장 우수한 답변(또는 첫 답변)을 선택하는 방식.
    * "우수" 판단은 score_fn으로 할 수도 있고, default로 첫 번째만 택할 수도 있음.
    """

    def __init__(self, model: BaseModel = None, n: int = 5, score_fn="heegyu/ko-reward-model-1.3b-v0.1", **kwargs):
        super().__init__(model=model, **kwargs)

        self.n = n
        self.score_fn = score_fn
        
        self.reward_model = pipeline("text-classification", model=self.score_fn)
        

    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        data 내 각 샘플에 대해:
          - N번 반복하여 후보 생성
          - score_fn이 있으면 최고 점수 후보 선택, 없으면 첫 후보
        """
        if self.model is None:
            raise ValueError("BestOfN requires a 'model' instance.")
        
        for sample in data:
            prompt = sample["input"]
            # N번 후보 생성 (batch 호출로 최적화 가능)
            candidates = []
            for _ in range(self.n):
                # generate_batch는 리스트를 반환하므로 [0]만 취함
                outputs = self.model.generate_batch([sample])  
                # 예: [{"input":..., "reference":..., "prediction":"..."}]
                candidate_text = outputs[0]["prediction"]
                candidates.append(candidate_text)
                

            # 점수 계산하여 베스트 후보 선택
            if self.score_fn is not None:

                score_list=[]

                for candidate_text in candidates:
                    """
                    scores = [self.score_fn(candidate, sample) for candidate in candidates]
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    """
                    template=f"""<human>:
                    {prompt}
                    <bot>:
                    {candidate_text}
                    <|endoftext|>
                    """
                    score=float(self.reward_model(template)[0]["score"])
                    score_list.append(score)
                    
                    

                best_idx=max(range(len(score_list)), key=lambda i: float(score_list[i]))

            else:
                # default: 첫 번째 후보
                best_idx = 0
            
            sample["prediction"] = candidates[best_idx]

        return data
