import argparse
import json
from typing import Any, Dict, List, Optional, Union
from llm_eval.runner import PipelineRunner

def _parse_json_str(json_str: Optional[str]) -> Dict[str, Any]:
    """
    간단한 헬퍼 함수:
      - None이 아니면 JSON으로 파싱하여 dict 반환
      - 예외 시에는 {}를 반환
    """
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse JSON string: {json_str}")
        print(e)
        return {}

class Evaluator:
    """
    사용자가 가장 단순하게 사용할 수 있는 고수준 Evaluator 클래스.
    
    예시:
        >>> import evaluator as ev
        >>> e = ev.Evaluator()
        >>> result = e.run(
        ...     model="huggingface",
        ...     dataset="haerae_bench",
        ...     subset="csat_geo",
        ...     evaluation_method="string_match"
        ... )
        >>> print(result["metrics"])
    """

    def __init__(
        self,
        default_model_backend: str = "huggingface",
        default_scaling_method: Optional[str] = None,
        default_evaluation_method: str = "string_match",
        default_split: str = "test",
    ):
        """
        Args:
            default_model_backend (str): 모델 백엔드 기본값
            default_scaling_method (str | None): 스케일링 방법 기본값 (없으면 직접 지정)
            default_evaluation_method (str): 평가 방법 기본값
            default_split (str): 데이터셋 split 기본값
        """
        self.default_model_backend = default_model_backend
        self.default_scaling_method = default_scaling_method
        self.default_evaluation_method = default_evaluation_method
        self.default_split = default_split

    def run(
        self,
        model: Optional[str] = None,
        dataset: str = "haerae_bench",
        subset: Union[str, List[str], None] = None,
        split: Optional[str] = None,
        scaling_method: Optional[str] = None,
        evaluation_method: Optional[str] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        scaling_params: Optional[Dict[str, Any]] = None,
        evaluator_params: Optional[Dict[str, Any]] = None,
        
        
        device:str=None,
        model_name:str=None

    ) -> Dict[str, Any]:
        """
        실제로 LLM 평가 파이프라인을 실행하는 함수.
        
        Args:
            model (str): 모델 백엔드 식별자 (Registry에 등록된 이름. 예: "huggingface", "openai", "multi")
            dataset (str): 데이터셋 식별자 (Registry에 등록된 이름. 예: "haerae_bench")
            subset (str | list[str] | None): 데이터셋 하위 태스크
            split (str): "train"/"valid"/"test" 등
            scaling_method (str): 스케일링(디코딩) 방법 (self-consistency, Best-of-N 등)
            evaluation_method (str): 평가 방법 (string_match, llm-as-a-judge 등)
            dataset_params (dict): dataset 로더 초기화 파라미터
            model_params (dict): 모델(백엔드) 초기화 파라미터
            scaling_params (dict): 스케일링 방법 초기화 파라미터
            evaluator_params (dict): 평가 메서드 초기화 파라미터
        
        Returns:
            Dict[str, Any]: 파이프라인 실행 결과(메트릭, 샘플별 출력 등)
        """
        model_backend_name = model or self.default_model_backend
        final_split = split if split is not None else self.default_split
        final_scaling = scaling_method or self.default_scaling_method
        final_eval = evaluation_method or self.default_evaluation_method

        runner = PipelineRunner(
            dataset_name=dataset,
            subset=subset,
            split=final_split,
            model_backend_name=model_backend_name,
            scaling_method_name=final_scaling,
            evaluation_method_name=final_eval,
            dataset_params=dataset_params or {},
            model_backend_params=model_params or {},
            scaling_params=scaling_params or {},
            evaluator_params=evaluator_params or {},
            device=device,
            model_name=model_name
        )
        results = runner.run()
        return results

def main():
    parser = argparse.ArgumentParser(description="Simple LLM Evaluator CLI")
    parser.add_argument("--model", type=str, default="huggingface", help="Model backend name (registry key)")
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="haerae_bench", help="Dataset name (registry key)")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train/valid/test)")
    parser.add_argument("--scaling_method", type=str, default="best_of_n", help="Scaling method name (registry key)")
    parser.add_argument("--evaluation_method", type=str, default="string_match", help="Evaluation method name")
    parser.add_argument("--output_file", type=str, default=None, help="Where to store results (JSON)")

    # 추가 파라미터를 JSON 문자열로 받도록 옵션 추가
    parser.add_argument("--dataset_params", type=str, default=None, help="JSON string for dataset params")
    parser.add_argument("--model_params", type=str, default=None, help="JSON string for model params")
    parser.add_argument("--scaling_params", type=str, default=None, help="JSON string for scaling params")
    parser.add_argument("--evaluator_params", type=str, default=None, help="JSON string for evaluator params")
    
    args = parser.parse_args()

    # JSON 파싱
    dataset_params = _parse_json_str(args.dataset_params)
    model_params = _parse_json_str(args.model_params)
    scaling_params = _parse_json_str(args.scaling_params)
    evaluator_params = _parse_json_str(args.evaluator_params)

    evaluator = Evaluator()
    results = evaluator.run(
        model=args.model,
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        scaling_method=args.scaling_method,
        evaluation_method=args.evaluation_method,
        dataset_params=dataset_params,
        model_params=model_params,
        scaling_params=scaling_params,
        evaluator_params=evaluator_params,

        device=args.device,
        model_name=args.model_name

    )

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
