from typing import List, Dict, Any, Optional, Union

class BaseModel:
    """
    (개발 용이를 위해 개발 단계에서는 한국어로 작성, 이후 영어로 대체)
    모든 모델 백엔드가 상속해야 할 추상 클래스.

    필수 구현 메서드:
      - generate_batch(self, inputs, return_logits=False) -> List[Dict[str, Any]]
        * inputs: [{"input": str, "reference": str, ...}, ...]
        * return_logits: True일 경우 logits이나 log_probs 정보를 추가로 반환할 수도 있음
        * 반환: [{"input":..., "reference":..., 
                  "prediction":...,        # 모델 최종 문자열 출력
                  "logits": (optional)..., # if return_logits=True
                  ...}, ...]
    """

    def __init__(self, **kwargs):
        # config, endpoint, 인증토큰 등 필요한 거 저장
        pass

    def generate_batch(
        self,
        inputs: List[Dict[str, Any]],
        return_logits: bool = False,
        cot: bool = False,
        batch_size: Optional[Union[int, str]] = "auto"
    ) -> List[Dict[str, Any]]:
        """
        LLM으로부터 텍스트(답안)를 생성하는 메서드.
        Args:
            inputs: [{"input": str, "reference": str, ...}, ...]
            return_logits: True일 경우, logits나 logprobs 등 추가 정보를 반환 가능.
            cot: True이면, 1) 모델 프롬프트에 'Let's think step by step' 등을 추가,
        2) 모델이 reasoning을 "chain_of_thought" 필드에 담아 반환할 수 있음.
        Returns:
            동일 리스트(또는 복사본)에 
            [
              {
                "input": ...,
                "reference": ...,
                "prediction": <생성된 답변>,
                "logits": (선택),
                "chain_of_thought": "...(중간 reasoning)..."
                ...
              },
              ...
            ]
        """
        raise NotImplementedError("Subclasses must implement generate_batch().")
    
    def generate_until(
        self,
        prompt: str,
        stop_sequences: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """특정 시퀀스가 나올 때까지 텍스트 생성
        
        Args:
            prompt (str): 입력 프롬프트 텍스트
            stop_sequences (List[str]): 생성을 중단할 시퀀스 목록
            max_tokens (Optional[int]): 생성할 최대 토큰 수
            **kwargs: 추가 생성 파라미터

        Returns:
            str: 생성된 텍스트
        """
        raise NotImplementedError
    
class BaseJudge:
    """
    Judge 모델(LLM-as-a-Judge) 기본 추상 클래스.
    '이미 생성된 텍스트(답변)'를 입력으로 받아, 품질/적합도를 평가하는 역할.
    예) Chain-of-Thought 기반 Self-Consistency 평가, 별점(1~5점) 등
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
                    - 보통 'prediction'(생성된 답안)을 활용해 품질 평가를 진행
        Returns:
            [{"judge_score": float or int, "judge_explanation": str, ...}, ...]
            - 각 샘플에 대해 평가 점수/판단을 추가해서 반환
        """
        raise NotImplementedError("Subclasses must implement judge_batch().")

class BaseRewardModel:
    """
    Reward 모델(DVTS 등에 활용 가능) 전용 추상 클래스.
    '문자열 답안' -> '스칼라 보상 값'을 추정하는 역할.
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
                    - 보통 'prediction'을 인풋 삼아, 보상 점수를 산출
        Returns:
            [{"reward": float, ...}, ...]
            - 각 샘플에 'reward' 필드를 추가
        """
        raise NotImplementedError("Subclasses must implement score_batch().")
