from typing import Dict, Type
from .base import BaseModel

# 1) model들을 등록할 전역 레지스트리 (dict)
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


# 2) 레지스트리에 등록할 헬퍼 함수
def register_model(name: str):
    """
    Model /Judge/ Reward 클래스를 레지스트리에 등록하기 위한 데코레이터.
    사용 예:
        @register_model("vllm")
        class VLLMModel(BaseModel):
            ...
    """

    def decorator(cls: Type[BaseModel]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator

# 3) 레지스트리에서 model 인스턴스를 생성하는 함수
def load_model(name: str, device,model_name,**kwargs) -> BaseModel:
    """
    문자열 name을 받아 해당 모델 클래스를 찾아 인스턴스화 후 반환.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Please register it in MODEL_REGISTRY."
        )
    model_cls = MODEL_REGISTRY[name]
    return model_cls(model_name_or_path=model_name,device=device,**kwargs)


# 5) 실제 backend들 import -> 데코레이터로 등록
# from .vllm_backend import VLLMModel
# from .huggingface_backend import HFModel
from .openai_backend import OpenAIModel
from .multi import MultiModel
from .huggingface import HuggingFaceModel