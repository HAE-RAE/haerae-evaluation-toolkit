import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value, Sequence
from typing import Optional, List, Dict, Any

from llm_eval.datasets.base import BaseDataset
from llm_eval.datasets import register_dataset
from llm_eval.utils.logging import get_logger

logger = get_logger(name="benchhub_dataset", level="INFO")

# --- Helper Functions ---
def _fix_value_to_str(value: Any) -> str:
    """Helper function to convert any value to a string, handling lists and None."""
    if isinstance(value, list):
        return " | ".join(map(str, value))
    if pd.isna(value) or value is None:
        return ""
    return str(value)

def _ensure_list_of_str(value: Any) -> List[str]:
    """Helper function to ensure the value is a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        # Check for empty list or list with None values
        if len(value) == 0:
            return []
        return [_fix_value_to_str(v) for v in value if v is not None]
    # For scalar values, check if it's NaN
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        # If pd.isna() fails (e.g., for certain objects), continue
        pass
    if isinstance(value, str):
        # Handle pipe-separated strings
        return [v.strip() for v in value.split("|") if v.strip()]
    return [_fix_value_to_str(value)]


def _contains_any_filter(value: Any, filters: Optional[List[str]]) -> bool:
    """
    Checks if the value (or any element if it's a list) contains any of the filter strings.
    Filter matching is case-insensitive.
    """
    if not filters:  # No filters means the item passes
        return True
    
    normalized_filters = [f.lower() for f in filters]
    
    if isinstance(value, list):
        # For lists, check if any element contains any filter string
        for item_val_in_list in value:
            item_str = _fix_value_to_str(item_val_in_list).lower()
            if any(f_val in item_str for f_val in normalized_filters):
                return True
        return False # No element in the list matched any filter
    else:
        # For single values, convert to string and check
        val_str = _fix_value_to_str(value).lower()
        return any(f_val in val_str for f_val in normalized_filters)

@register_dataset("benchhub")
class BenchHubDataset(BaseDataset):
    """
    BenchHub Dataset Loader for HAE-RAE Evaluation Toolkit.

    This class loads data from the 'BenchHub/BenchHub-Ko' or 'BenchHub/BenchHub-En'
    repositories on Hugging Face Hub. It supports chunk-based loading and
    flexible filtering based on various metadata columns such as problem_type,
    task_type, target_type, subject_type, and benchmark_name.

    The output format is a list of dictionaries, where each dictionary
    represents a sample and includes "input", "reference", "options",
    "_subset_name", and "metadata" keys, compatible with the HRET pipeline.
    """

    EXPECTED_FEATURES = Features({
        'problem_type': Value(dtype='string'),
        'context': Value(dtype='string'), 
        'prompt': Value(dtype='string'),
        'options': Sequence(Value(dtype='string')), 
        'answer': Value(dtype='string'),
        'reference': Value(dtype='string'), 
        'task_type': Value(dtype='string'),
        'target_type': Value(dtype='string'),
        'subject_type': Sequence(Value(dtype='string')), 
        'benchmark_name': Value(dtype='string'),
        'mcqa_meta': Value(dtype='string'), 
        'original_category': Value(dtype='string'),
        'additional_info': Value(dtype='string'),
        'split': Value(dtype='string'),
    })


    def __init__(
        self,
        dataset_name: str = "benchhub",
        split: str = "train",
        language: str = "ko",
        base_prompt_template: Optional[str] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
        problem_types: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        target_types: Optional[List[str]] = None,
        subject_types: Optional[List[str]] = None,
        benchmark_names: Optional[List[str]] = None,
        chunk_size: int = 1000,
        hf_load_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs 
    ):
        # Remove duplicate arguments from kwargs to prevent conflicts
        kwargs.pop('subset', None)
        kwargs.pop('split', None)
        
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            subset=None, 
            base_prompt_template=base_prompt_template,
            **kwargs
        )
        
        self.language = language.lower()
        if self.language not in ["ko", "en"]:
            raise ValueError("Language must be 'ko' or 'en'.")
        self.repo_id = f"BenchHub/BenchHub-{ 'Ko' if self.language == 'ko' else 'En' }"
        
        self.sample_size = sample_size
        self.seed = seed
        self.filter_problem_types = problem_types
        self.filter_task_types = task_types
        self.filter_target_types = target_types
        self.filter_subject_types = subject_types
        self.filter_benchmark_names = benchmark_names
        self.chunk_size = chunk_size
        
        self.hf_load_kwargs = hf_load_kwargs if hf_load_kwargs else {}
        self.hf_load_kwargs.setdefault('name', 'default')


    def load(self) -> List[Dict[str, Any]]:
        logger.info(
            f"Loading BenchHub dataset: {self.repo_id}, split: {self.split}, language: {self.language}"
        )
        logger.info(f"Applied filters: problem_types={self.filter_problem_types}, "
                    f"task_types={self.filter_task_types}, target_types={self.filter_target_types}, "
                    f"subject_types={self.filter_subject_types}, benchmark_names={self.filter_benchmark_names}")

        all_filtered_data = []
        processed_samples_count = 0

        try:
            ds_info = load_dataset(self.repo_id, name=self.hf_load_kwargs.get('name'), split=self.split, **{k:v for k,v in self.hf_load_kwargs.items() if k!='name'})
            total_samples = len(ds_info)
            logger.info(f"Total samples in '{self.split}' split: {total_samples}")
            use_streaming_approach = False
        except Exception as e:
            logger.warning(
                f"Could not determine total samples for split '{self.split}' directly (Error: {e}). "
                f"Proceeding with chunked loading. Progress bar may not show total."
            )
            total_samples = float('inf') 
            use_streaming_approach = True

        if not use_streaming_approach and total_samples > 0:
            for i in range(0, total_samples, self.chunk_size):
                chunk_split = f"{self.split}[{i}:{min(i + self.chunk_size, total_samples)}]"
                try:
                    ds_chunk = load_dataset(self.repo_id, name=self.hf_load_kwargs.get('name'), split=chunk_split, **{k:v for k,v in self.hf_load_kwargs.items() if k!='name'})

                except Exception as e_chunk:
                    import traceback
                    logger.error(f"Error loading chunk {chunk_split}: {e_chunk}. Skipping this chunk.")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
                
                if ds_chunk:
                    filtered_chunk = self._filter_dataset_object(ds_chunk)
                    all_filtered_data.extend(self._convert_to_hret_format(filtered_chunk))
                    processed_samples_count += len(ds_chunk)
                    logger.debug(f"Processed {len(ds_chunk)} samples in chunk {chunk_split}. Filtered down to {len(filtered_chunk)}. Total HRET samples so far: {len(all_filtered_data)}")
        

        elif use_streaming_approach or total_samples == 0:
            current_pos = 0
            while True:
                chunk_split = f"{self.split}[{current_pos}:{current_pos + self.chunk_size}]"
                logger.debug(f"Attempting to load chunk: {chunk_split}")
                try:
                    ds_chunk = load_dataset(self.repo_id, name=self.hf_load_kwargs.get('name'), split=chunk_split, **{k:v for k,v in self.hf_load_kwargs.items() if k!='name'})
                except Exception as e_chunk:
                    import traceback
                    logger.info(f"Finished loading chunks or error in chunk {chunk_split}: {e_chunk}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    break 
                
                if not ds_chunk or len(ds_chunk) == 0:
                    logger.info("Empty chunk received, assuming end of dataset.")
                    break
                
                filtered_chunk = self._filter_dataset_object(ds_chunk)
                all_filtered_data.extend(self._convert_to_hret_format(filtered_chunk))
                processed_samples_count += len(ds_chunk)
                logger.debug(f"Processed {len(ds_chunk)} samples in chunk. Filtered down to {len(filtered_chunk)}. Total HRET samples so far: {len(all_filtered_data)}")
                
                if len(ds_chunk) < self.chunk_size:
                    logger.info("Loaded a partial chunk, assuming end of dataset.")
                    break
                current_pos += self.chunk_size
        
        logger.info(f"Finished loading. Total HRET formatted samples: {len(all_filtered_data)} from {processed_samples_count} scanned BenchHub samples.")
        
        # Apply sample_size and seed if specified
        if self.sample_size is not None and self.sample_size < len(all_filtered_data):
            import random
            if self.seed is not None:
                random.seed(self.seed)
            all_filtered_data = random.sample(all_filtered_data, self.sample_size)
            logger.info(f"Applied sample_size={self.sample_size}. Final sample count: {len(all_filtered_data)}")
        
        return all_filtered_data

    def _filter_dataset_object(self, dataset_obj: Dataset) -> Dataset:
        """Applies all configured filters to a datasets.Dataset object."""
        if self.filter_problem_types:
            dataset_obj = dataset_obj.filter(lambda x: _contains_any_filter(x.get('problem_type'), self.filter_problem_types))
        if self.filter_task_types:
            dataset_obj = dataset_obj.filter(lambda x: _contains_any_filter(x.get('task_type'), self.filter_task_types))
        if self.filter_target_types:
            dataset_obj = dataset_obj.filter(lambda x: _contains_any_filter(x.get('target_type'), self.filter_target_types))
        if self.filter_subject_types:
            dataset_obj = dataset_obj.filter(lambda x: _contains_any_filter(x.get('subject_type'), self.filter_subject_types))
        if self.filter_benchmark_names:
            dataset_obj = dataset_obj.filter(lambda x: _contains_any_filter(x.get('benchmark_name'), self.filter_benchmark_names))
        return dataset_obj

    def _convert_to_hret_format(self, dataset_obj: Dataset) -> List[Dict[str, Any]]:
        """Converts a filtered datasets.Dataset object to HRET's list of dicts format."""
        hret_list = []
        for item in dataset_obj:
            input_text_parts = []
            context = _fix_value_to_str(item.get("context"))
            prompt = _fix_value_to_str(item.get("prompt"))
            
            if context:
                input_text_parts.append(context)
            input_text_parts.append(prompt)
            query_for_template = "\n".join(input_text_parts)

            if self.base_prompt_template:
                try:
                    format_args = {k: _fix_value_to_str(v) for k, v in item.items()}
                    format_args["query"] = query_for_template
                    # Add options_str for MCQA templates
                    options = _ensure_list_of_str(item.get("options"))
                    if options:
                        options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
                        format_args["options_str"] = options_str
                    hret_input = self.base_prompt_template.format_map(format_args)
                except KeyError as e:
                    logger.warning(f"Prompt template key {e} not found in item. Using raw query for input. Item keys: {list(item.keys())}")
                    hret_input = query_for_template 
            else:
                hret_input = query_for_template

            # Prioritize answer_str, then answer, then reference
            hret_reference = _fix_value_to_str(
                item.get("answer_str") or item.get("answer") or item.get("reference")
            )

            hret_options = _ensure_list_of_str(item.get("options"))
            
            metadata = {
                "problem_type": _fix_value_to_str(item.get("problem_type")),
                "task_type": _fix_value_to_str(item.get("task_type")),
                "target_type": _fix_value_to_str(item.get("target_type")),
                "subject_type": _ensure_list_of_str(item.get("subject_type")),
                "benchmark_name": _fix_value_to_str(item.get("benchmark_name")),
                "mcqa_meta": _fix_value_to_str(item.get("mcqa_meta")),
                "original_category": _fix_value_to_str(item.get("original_category")),
                "additional_info": _fix_value_to_str(item.get("additional_info")),
                "original_split": _fix_value_to_str(item.get("split")),
                "language": self.language,
                "original_benchhub_reference": _fix_value_to_str(item.get("reference")) 
            }
            
            hret_sample = {
                "input": hret_input,
                "reference": hret_reference,
                "options": hret_options,
                "_subset_name": _fix_value_to_str(item.get("benchmark_name")) or "benchhub_default_subset",
                "metadata": metadata
            }
            hret_list.append(hret_sample)
        return hret_list

    def info(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "repo_id": self.repo_id,
            "split": self.split,
            "language": self.language,
            "description": "BenchHub dataset loader. Aggregates various benchmarks with detailed classifications.",
            "filters_applied": {
                "problem_types": self.filter_problem_types,
                "task_types": self.filter_task_types,
                "target_types": self.filter_target_types,
                "subject_types": self.filter_subject_types,
                "benchmark_names": self.filter_benchmark_names,
            },
            "huggingface_load_kwargs": self.hf_load_kwargs,
            "evaluation_only": None 
        }
