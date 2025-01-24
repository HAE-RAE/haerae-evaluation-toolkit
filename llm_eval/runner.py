import logging
import time
from typing import Any, Dict, List, Optional, Union

from llm_eval.datasets import load_datasets, BaseDataset
from llm_eval.models import load_model, BaseModel
from llm_eval.scaling_methods import load_scaling_method, BaseScalingMethod
from llm_eval.evaluation import get_evaluator, BaseEvaluator
from llm_eval.utils.logging import get_logger

logger = get_logger(name="runner", level=logging.INFO)


class PipelineRunner:
    """
    전체 LLM 평가 파이프라인을 캡슐화하는 Runner 클래스.
    'dataset_name/subset/split' -> 데이터셋 Load
    (참고) dataset.info()에서 "evaluation_only", "scaling_only" 필드를 확인하여 
            허용되지 않은 방법을 선택하면 에러 또는 경고
    'model_backend_name' -> 모델 백엔드
    'scaling_method_name' -> 스케일링 
    'evaluation_method_name' -> 평가 

    """

    def __init__(
        self,
        dataset_name: str,
        subset: Union[str, List[str], None] = None,
        split: str = "test",
        model_backend_name: str = "huggingface",
        scaling_method_name: Optional[str] = None,
        evaluation_method_name: str = "string_match",

        dataset_params: Optional[Dict[str, Any]] = None,
        model_backend_params: Optional[Dict[str, Any]] = None,
        scaling_params: Optional[Dict[str, Any]] = None,
        evaluator_params: Optional[Dict[str, Any]] = None,


        device : str = None,
        model_name: str = None,
    ):
        """
        Args:
            dataset_name (str): 데이터셋 식별자
            subset (str | list[str] | None): 하위 태스크(예: "csat_geo")
            split (str): "train"/"valid"/"test" 등
            model_backend_name (str): 모델 백엔드 식별자 ("huggingface", "openai", "multi" 등)
            scaling_method_name (str | None): 스케일링(디코딩) 방법 식별자
            evaluation_method_name (str): 평가 방법 식별자

            dataset_params (dict): Dataset 로더에 전달할 추가 파라미터 (HF config, 토큰 등)
            model_backend_params (dict): 모델 백엔드 초기화 시 사용할 파라미터 (endpoint, API key 등)
            scaling_params (dict): 스케일링 메서드 초기화 시 파라미터 (beam_size, n 등)
            evaluator_params (dict): evaluator 초기화 시 필요한 파라미터
        """
        
        self.device=device 
        self.model_name=model_name

        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.model_backend_name = model_backend_name
        self.scaling_method_name = scaling_method_name
        self.evaluation_method_name = evaluation_method_name

        self.dataset_params = dataset_params or {}
        self.model_backend_params = model_backend_params or {}
        self.scaling_params = scaling_params or {}
        self.evaluator_params = evaluator_params or {}

        self.dataset: Optional[BaseDataset] = None
        self.model: Optional[BaseModel] = None
        self.scaler: Optional[BaseScalingMethod] = None
        self.evaluator: Optional[BaseEvaluator] = None

        # 사전 로드/초기화 (lazy-load가 나을지도..?)
        self._load_components()

    def _load_components(self) -> None:
        """
        내부적으로 dataset, model, scaler, evaluator를 로드하고,
        dataset.info()를 통해 'scaling_only', 'evaluation_only' 같은 제약을 확인함.
        """
        # 1) Dataset
        logger.info(f"[Pipeline] Loading dataset: {self.dataset_name}, subset={self.subset}, split={self.split}")
        self.dataset = load_datasets(
            name=self.dataset_name,
            subset=self.subset,
            split=self.split,
            **self.dataset_params  # dataset_params 적용
        )

        ds_info = self.dataset.info()
        # ds_info 내부에 { ..., "scaling_only": [...], "evaluation_only": [...], ... } 등이 있을 수 있음
        scaling_only = ds_info.get("scaling_only", None)       # 예: ["beam_search", "best_of_n"]
        evaluation_only = ds_info.get("evaluation_only", None) # 예: ["llm_judge", "string_match"]

        # 2) Model
        logger.info(f"[Pipeline] Loading model backend: {self.model_backend_name} with {self.model_backend_params}")
        self.model = load_model(self.model_backend_name, self.device, self.model_name, **self.model_backend_params)

        # 3) Scaling Method (optional)
        if self.scaling_method_name:
            if scaling_only is not None:
                # 데이터셋에서 특정 스케일링만 허용한다고 명시된 경우
                if self.scaling_method_name not in scaling_only:
                    raise ValueError(
                        f"Dataset '{self.dataset_name}' only allows scaling methods {scaling_only}, "
                        f"but got '{self.scaling_method_name}'."
                    )
            logger.info(f"[Pipeline] Loading scaling method: {self.scaling_method_name} with {self.scaling_params}")
            self.scaler = load_scaling_method(
                self.scaling_method_name,
                model=self.model,  # 모델 인스턴스 주입
                **self.scaling_params
            )
        else:
            if scaling_only is not None:
                # 데이터셋이 특정 스케일링만 허용한다고 했는데, None이면 => 에러 or 경고
                raise ValueError(
                    f"Dataset '{self.dataset_name}' requires a scaling method from {scaling_only}, "
                    f"but scaling_method_name=None."
                )

        # 4) Evaluator
        if evaluation_only is not None:
            # 데이터셋이 특정 evaluation만 허용
            if self.evaluation_method_name not in evaluation_only:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' only allows evaluation methods {evaluation_only}, "
                    f"but got '{self.evaluation_method_name}'."
                )

        logger.info(f"[Pipeline] Loading evaluator: {self.evaluation_method_name} with {self.evaluator_params}")
        self.evaluator = get_evaluator(self.evaluation_method_name, **self.evaluator_params)

    def run(self) -> Dict[str, Any]:
        """
        실제 파이프라인 실행: 
            - Dataset load -> (Scaling) Inference -> Evaluation -> 결과 반환
        Returns:
            {
                "metrics": {...},
                "samples": [...],
                "info": {...}  # (선택) 부가적인 실행 정보
            }
        """
        if not self.dataset or not self.model or not self.evaluator:
            raise RuntimeError("Pipeline components are not fully loaded.")

        start_time = time.time()

        # 1) Dataset load
        data = self.dataset.load()  # [{"input":..., "reference":..., ...}, ...]
        logger.info(f"[Pipeline] Dataset loaded. # of samples: {len(data)}")

        # 2) Scaling / Inference
        if self.scaler:
            logger.info(f"[Pipeline] Applying scaling method: {self.scaling_method_name}")
            predictions = self.scaler.apply(data)  
        else:
            logger.info("[Pipeline] Direct model_backend inference (no scaling).")
            predictions = self.model.generate_batch(data)
        logger.info(f"[Pipeline] Inference done for {len(predictions)} items.")

        # 3) Evaluation
        logger.info(f"[Pipeline] Evaluating with {self.evaluation_method_name}")
        eval_results = self.evaluator.evaluate(predictions, model=None)
        # eval_results 예시: {"metrics": {...}, "samples": [...], ...}

        # 4) 결과 요약
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[Pipeline] Done. Elapsed: {elapsed:.2f} sec")

        # 추가 정보
        eval_results["info"] = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "model_backend_name": self.model_backend_name,
            "scaling_method_name": self.scaling_method_name,
            "evaluation_method_name": self.evaluation_method_name,
            "elapsed_time_sec": elapsed,
        }

        return eval_results
