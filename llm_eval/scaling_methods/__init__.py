from typing import Dict, Type
from .base import BaseScalingMethod

# 1) scaling method 들을 등록할 전역 레지스트리 (dict)
SCALING_REGISTRY: Dict[str, Type[BaseScalingMethod]] = {}

# 2) 레지스트리에 등록할 헬퍼 함수
def register_scaling_method(name: str):
    """
    ScalingMethod 클래스를 레지스트리에 등록하기 위한 데코레이터.
    사용 예:
        @register_scaling_method("best_of_n")
        class BestOfN(BaseScalingMethod):
            ...
    """
    def decorator(cls: Type[BaseScalingMethod]):
        if name in SCALING_REGISTRY:
            raise ValueError(f"Scaling method '{name}' already registered.")
        SCALING_REGISTRY[name] = cls
        return cls
    return decorator

# 3) 레지스트리에서 ScalingMethod 인스턴스를 생성하는 헬퍼
def load_scaling_method(name: str, **kwargs) -> BaseScalingMethod:
    """
    문자열 name으로 레지스트리에서 ScalingMethod 클래스를 찾아 인스턴스화해 반환.
    """
    if name not in SCALING_REGISTRY:
        raise ValueError(f"Unknown scaling method: {name}. Available: {list(SCALING_REGISTRY.keys())}")
    scaling_cls = SCALING_REGISTRY[name]
    return scaling_cls(**kwargs)

# 4) import -> 데코레이터로 등록
from .self_consistency import SelfConsistencyScalingMethod
from .best_of_n import BestOfN
# from .beam_search import BeamSearch

