from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from .base import BaseDataset
from . import register_dataset

@register_dataset("k2_eval")
class K2_EvalDataset(BaseDataset):
    """
    k2_eval 데이터셋 클래스.

    - subset: None -> 전체 로드
              list[str] -> 해당 여러 subset만 로드하여 합침
              str -> 해당 subset만 로드
    - 각 샘플에는 'options' 필드를 포함 (원 데이터에 없으면 ["(A)", "(B)", "(C)", "(D)", "(E)"] 사용)
    - '_subset_name' 필드를 추가해, 어떤 서브태스크에서 불러온 데이터인지 표시
      (Evaluation 시점에 subset별 점수를 구분하기 위해)

    사용 예시:
        ds = HaeraeDataset(
            dataset_name="haerae_bench",
            subset=["csat_geo", "csat_law"],  # 여러 subset
            split="test"
        )
        data = ds.load()
        # data -> [
        #   {
        #     "input": "...",
        #     "reference": "...",
        #     "options": ["(A)","(B)","(C)","(D)","(E)"], # log-prob 계산을 위해 필요
        #     "_subset_name": "csat_geo"
        #   }, ...
        # ]
    """
    def __init__(
        self, 
        dataset_name: str = "HAERAE-HUB/K2-Eval",
        subset: Optional[Union[str, list]] = None,
        split: str = "test",
        **kwargs
    ):
        super().__init__(dataset_name, split=split, subset=subset, **kwargs)
        
    def load(self) -> List[Dict[str, Any]]:
        """
        데이터를 로드하고 [{"input":..., "reference":..., "options":..., "_subset_name":...}, ...] 형태를 반환.
        """
        # 1) subset 유형에 따라 나눔
        if self.subset is None:
            self.subset = ['knowledge']


        if isinstance(self.subset, list):
            # 여러 다수의 subset인 경우
            all_items = []
            for sub in self.subset:
                partial_data = load_dataset(
                    self.dataset_name,
                    sub,
                    split=self.split,
                    **self.kwargs
                )
                all_items.extend(self._convert_to_list(partial_data, subset_name=sub))
            return all_items

        else:  # subset이 문자열 1개
            raw_data = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                **self.kwargs
            )
            return self._convert_to_list(raw_data, subset_name=self.subset)

    def _convert_to_list(self, hf_dataset, subset_name: str) -> List[Dict[str, Any]]:
        """
        HuggingFace Dataset 객체(hf_dataset)를 순회하며,
        {"input":..., "reference":..., "options":..., "_subset_name": subset_name} 형태로 변환.
        """
        processed_list = []
        # 고정 선택지 A~E
        options = ["(A)", "(B)", "(C)", "(D)"]

        for item in hf_dataset:
            # 원본 데이터에서 query, answer 필드 추출 (없으면 빈 문자열)
            query = item.get("question", "")
            answer = item.get("gold_answer", "")
            processed_list.append({
                "input": query.strip(),
                "reference": answer, # 숫자 reference는 strip() 하면 안됨
                "options": options,
                "_subset_name": subset_name,
            })
        return processed_list

    def get_raw_samples(self) -> Any:
        """
        원본 데이터를 반환.
        여러 subset이면 리스트로, 하나거나 None이면 단일 Dataset 반환 (간단 예시).
        """
        if self.subset is None:
            return load_dataset(self.dataset_name, split=self.split, **self.kwargs)
        elif isinstance(self.subset, list):
            result = []
            for s in self.subset:
                partial = load_dataset(
                    self.dataset_name, s, split=self.split, **self.kwargs
                )
                result.append(partial)
            return result
        else:
            return load_dataset(
                self.dataset_name, 
                self.subset,
                split=self.split,
                **self.kwargs
            )

    def info(self) -> Dict[str, Any]:
        """
        데이터셋 관련 메타 정보.
        """
        return {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
            "description": (
                "K2_Eval benchmark featuring 90 handwritten instructions that require in-depth knowledge of Korean language and culture for accurate completion. https://huggingface.co/datasets/HAERAE-HUB/K2-Eval"
                "subset=list -> load partial subsets, "
                "subset=str -> load single subset."
            )
        }
