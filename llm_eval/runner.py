#!/usr/bin/env python
"""
PipelineRunner 

This module encapsulates the entire LLM evaluation pipeline:
  1. Load a dataset (with optional subset and split)
  2. Load a model backend
  3. Optionally apply a scaling method for model inference
  4. Evaluate the generated predictions using an evaluator
  5. Optionally apply a language penalizer to each prediction, using a parameterized target language
  6. Aggregate metrics and sample outputs into an EvaluationResult

This runner can be used via CLI or integrated into an API.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

from llm_eval.datasets import load_datasets, BaseDataset
from llm_eval.models import load_model, BaseModel
from llm_eval.scaling_methods import load_scaling_method, BaseScalingMethod
from llm_eval.evaluation import get_evaluator, BaseEvaluator
from llm_eval.utils.logging import get_logger
from llm_eval.utils.util import EvaluationResult
from llm_eval.internal.benchhub_info import DATASETS as BENCHHUB_INFO_ENTRIES
from llm_eval.utils.prompt_template import (
    format_few_shot_prompt_prefix, 
    DEFAULT_FEW_SHOT_INSTRUCTION, 
    DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE,
)
from llm_eval.utils.metrics import language_penalizer

logger = get_logger(name="runner", level=logging.INFO)


@dataclass
class PipelineConfig:
    """Configuration class for the pipeline runner."""
    dataset_name: str
    subset: Union[str, List[str], None] = None
    split: str = "test"
    model_backend_name: str = "huggingface"
    scaling_method_name: Optional[str] = None
    evaluation_method_name: str = "string_match"
    dataset_params: Optional[Dict[str, Any]] = None
    model_backend_params: Optional[Dict[str, Any]] = None
    scaling_params: Optional[Dict[str, Any]] = None
    evaluator_params: Optional[Dict[str, Any]] = None
    language_penalize: bool = True
    target_lang: str = "ko"
    custom_cot_parser: Optional[Callable[[str], Tuple[str, str]]] = None
    num_few_shot: int = 0
    few_shot_split: Optional[str] = None
    few_shot_instruction: str = DEFAULT_FEW_SHOT_INSTRUCTION
    few_shot_example_template: str = DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        self.dataset_params = self.dataset_params or {}
        self.model_backend_params = self.model_backend_params or {}
        self.scaling_params = self.scaling_params or {}
        self.evaluator_params = self.evaluator_params or {}
        
        if not isinstance(self.num_few_shot, int) or self.num_few_shot < 0:
            logger.warning(f"Invalid num_few_shot value: {self.num_few_shot}. Setting to 0.")
            self.num_few_shot = 0

    def tracking_metadata(self) -> Dict[str, Any]:
        """Fields suitable for saved results and external experiment trackers."""
        metadata = {
            key: getattr(self, key)
            for key in (
                "dataset_name", "subset", "split", "model_backend_name",
                "scaling_method_name", "evaluation_method_name", "num_few_shot",
                "few_shot_split", "language_penalize", "target_lang",
            )
        }

        # Preserve numeric experiment settings without serializing provider-specific
        # credentials, headers, custom callables, or arbitrary user dictionaries.
        allowed_params = {
            "dataset_params": ("seed", "sample_size"),
            "model_backend_params": ("temperature", "top_p", "batch_size", "max_new_tokens", "max_tokens", "do_sample"),
            "scaling_params": ("n", "n_paths", "beam_size", "batch_size"),
            "evaluator_params": ("ignore_case", "ignore_numbers", "ignore_punctuation", "mcqa"),
        }
        for group, keys in allowed_params.items():
            params = getattr(self, group)
            metadata[group] = {key: params[key] for key in keys if isinstance(params.get(key), (int, float))}
        return metadata


class ComponentManager:
    """Manages loading and validation of pipeline components."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset: Optional[BaseDataset] = None
        self.model: Optional[BaseModel] = None
        self.scaler: Optional[BaseScalingMethod] = None
        self.evaluator: Optional[BaseEvaluator] = None

    def load_all_components(self) -> None:
        """Load all pipeline components in the correct order."""
        self._load_dataset()
        self._load_model()
        self._load_scaling_method()
        self._load_evaluator()

    def _load_dataset(self) -> None:
        """Load the dataset component."""
        logger.info(f"[Pipeline] Loading dataset: {self.config.dataset_name}, "
                   f"subset={self.config.subset}, split={self.config.split}")
        self.dataset = load_datasets(
            name=self.config.dataset_name,
            subset=self.config.subset,
            split=self.config.split,
            **self.config.dataset_params
        )

    def _load_model(self) -> None:
        """Load the model backend component."""
        # Inject custom CoT parser if provided
        model_params = self.config.model_backend_params.copy()
        if self.config.custom_cot_parser is not None:
            model_params["cot_parser"] = self.config.custom_cot_parser

        logger.info(f"[Pipeline] Loading model backend: {self.config.model_backend_name}")
        self.model = load_model(self.config.model_backend_name, **model_params)

    def _load_scaling_method(self) -> None:
        """Load the scaling method component if specified."""
        if not self.config.scaling_method_name:
            self._validate_no_scaling_required()
            return

        self._validate_scaling_method_allowed()
        logger.info(f"[Pipeline] Loading scaling method: {self.config.scaling_method_name}")
        self.scaler = load_scaling_method(
            self.config.scaling_method_name,
            model=self.model,
            **self.config.scaling_params
        )

    def _load_evaluator(self) -> None:
        """Load the evaluator component."""
        self._validate_evaluation_method_allowed()
        logger.info(f"[Pipeline] Loading evaluator: {self.config.evaluation_method_name}")
        
        if self.config.evaluation_method_name == "llm_judge":
            self.evaluator = get_evaluator(
                self.config.evaluation_method_name, 
                model=self.model, 
                **self.config.evaluator_params
            )
        else:
            self.evaluator = get_evaluator(
                self.config.evaluation_method_name, 
                **self.config.evaluator_params
            )

    def _validate_scaling_method_allowed(self) -> None:
        """Validate that the scaling method is allowed for this dataset."""
        ds_info = self.dataset.info()
        scaling_only = ds_info.get("scaling_only", None)
        
        if scaling_only is not None and self.config.scaling_method_name not in scaling_only:
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' only allows scaling methods {scaling_only}, "
                f"but got '{self.config.scaling_method_name}'."
            )

    def _validate_no_scaling_required(self) -> None:
        """Validate that no scaling method is required for this dataset."""
        ds_info = self.dataset.info()
        scaling_only = ds_info.get("scaling_only", None)
        
        if scaling_only is not None:
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' requires a scaling method from {scaling_only}, "
                f"but scaling_method_name is None."
            )

    def _validate_evaluation_method_allowed(self) -> None:
        """Validate that the evaluation method is allowed for this dataset."""
        ds_info = self.dataset.info()
        evaluation_only = ds_info.get("evaluation_only", None)
        
        if evaluation_only is None:
            return
            
        if isinstance(evaluation_only, bool) and evaluation_only:
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' requires a specific evaluation method, "
                f"but no specific methods were provided."
            )
        elif (not isinstance(evaluation_only, bool) and 
              self.config.evaluation_method_name not in evaluation_only):
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' only allows evaluation methods {evaluation_only}, "
                f"but got '{self.config.evaluation_method_name}'."
            )


class FewShotHandler:
    """Handles few-shot example preparation and processing."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config

    def prepare_few_shot_prefix(self) -> str:
        """Prepare few-shot prompt prefix if configured."""
        if self.config.num_few_shot <= 0:
            logger.info("[FewShot] num_few_shot is 0 or less. No few-shot examples will be prepared.")
            return ""

        few_shot_source_split = self.config.few_shot_split or self.config.split
        logger.info(f"[FewShot] Attempting to prepare {self.config.num_few_shot} few-shot examples "
                   f"from dataset '{self.config.dataset_name}', subset '{self.config.subset}', "
                   f"split '{few_shot_source_split}'.")

        try:
            return self._load_and_format_examples(few_shot_source_split)
        except Exception as e:
            logger.error(f"[FewShot] Critical error during loading or processing of few-shot examples "
                        f"from split '{few_shot_source_split}': {e}", exc_info=True)
            return ""

    def _load_and_format_examples(self, split: str) -> str:
        """Load and format few-shot examples from the specified split."""
        # Remove few-shot related parameters to prevent infinite recursion
        fs_dataset_params = self.config.dataset_params.copy()
        for key in ['num_few_shot', 'few_shot_split', 'few_shot_instruction', 'few_shot_example_template']:
            fs_dataset_params.pop(key, None)
        
        few_shot_ds_loader = load_datasets(
            name=self.config.dataset_name,
            subset=self.config.subset,
            split=split,
            **fs_dataset_params 
        )
        few_shot_examples_raw = few_shot_ds_loader.load()
        
        if not few_shot_examples_raw:
            logger.warning(f"[FewShot] No data found for few-shot examples from split '{split}'. "
                          f"Returning empty prefix.")
            return ""
        
        return self._format_examples(few_shot_examples_raw, split)

    def _format_examples(self, examples: List[Dict[str, Any]], split: str) -> str:
        """Format the loaded examples into a few-shot prefix."""
        num_available = len(examples)
        num_to_take = min(self.config.num_few_shot, num_available)

        if num_to_take == 0:
            logger.warning(f"[FewShot] Not enough samples available ({num_available}) in '{split}' "
                          f"to create {self.config.num_few_shot} few-shot examples. No prefix generated.")
            return ""
        
        if num_to_take < self.config.num_few_shot:
            logger.warning(f"[FewShot] Requested {self.config.num_few_shot} examples, "
                          f"but only {num_to_take} are available in '{split}'. Using {num_to_take} examples.")

        selected_samples = examples[:num_to_take]
        logger.info(f"[FewShot] Selected {len(selected_samples)} samples for few-shot prefix construction.")
        
        prefix_str = format_few_shot_prompt_prefix(
            selected_samples,
            instruction=self.config.few_shot_instruction,
            example_template=self.config.few_shot_example_template
        )
        
        if not prefix_str:
            logger.warning("[FewShot] Constructed few-shot prefix is empty. "
                          "This might happen if all selected samples were invalid for formatting.")
        else:
            logger.info(f"[FewShot] Successfully constructed few-shot prefix with {len(selected_samples)} examples.")
        
        return prefix_str

    def process_samples_for_inference(self, samples: List[Dict[str, Any]], 
                                    few_shot_prefix: str) -> List[Dict[str, Any]]:
        """Add few-shot prefix to samples and mark them appropriately."""
        processed_samples = []
        for sample in samples:
            new_sample = sample.copy()
            original_input = new_sample.get("input", "")
            new_sample["input"] = few_shot_prefix + original_input
            
            if self.config.num_few_shot > 0 and few_shot_prefix:
                new_sample["few_shot_applied"] = True
            
            processed_samples.append(new_sample)
        
        return processed_samples

    def filter_evaluation_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter samples for evaluation, excluding few-shot examples if from same split."""
        if (self.config.num_few_shot <= 0 or 
            (self.config.few_shot_split is not None and self.config.few_shot_split != self.config.split)):
            return samples

        if len(samples) <= self.config.num_few_shot:
            logger.warning(
                f"Number of samples in '{self.config.split}' ({len(samples)}) is less than or equal to "
                f"num_few_shot ({self.config.num_few_shot}). No samples left for evaluation."
            )
            return []
        
        logger.info(
            f"Using first {self.config.num_few_shot} samples from '{self.config.split}' for few-shot examples. "
            f"Evaluating on the remaining {len(samples) - self.config.num_few_shot} samples."
        )
        return samples[self.config.num_few_shot:]


class LanguagePenalizer:
    """Handles language penalization of predictions."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config

    def apply_penalization(self, eval_dict: Dict[str, Any]) -> None:
        """Apply language penalization to evaluation results if enabled."""
        if not self.config.language_penalize:
            return

        logger.info(f"Applying language penalizer with target language '{self.config.target_lang}'.")
        language_scores = []
        evaluated_samples = eval_dict.get("samples", [])
        
        for sample in evaluated_samples:
            pred_text = sample.get("original_prediction", sample.get("prediction", ""))
            
            if not isinstance(pred_text, str):
                logger.debug(f"Prediction text is not a string, cannot apply language penalizer. "
                           f"Sample ID: {sample.get('id', 'N/A')}, Type: {type(pred_text)}")
                lp_score = 0.0
            else:
                lp_score = language_penalizer(pred_text, target_lang=self.config.target_lang)
            
            sample["language_penalizer"] = lp_score
            language_scores.append(lp_score)
        
        self._update_metrics(eval_dict, language_scores)

    def _update_metrics(self, eval_dict: Dict[str, Any], language_scores: List[float]) -> None:
        """Update evaluation metrics with language penalizer scores."""
        eval_dict.setdefault("metrics", {})
        
        if language_scores:
            avg_lp = sum(language_scores) / len(language_scores)
            eval_dict["metrics"]["language_penalizer_average"] = avg_lp
            logger.info(f"Average language penalizer score: {avg_lp:.4f}")
        else:
            eval_dict["metrics"]["language_penalizer_average"] = 0.0
            logger.info("No samples to calculate language penalizer average or all predictions were non-string.")


class BenchHubInfoProcessor:
    """Processes BenchHub-specific information for evaluation results."""
    
    @staticmethod
    def add_benchmark_details(pipeline_info: Dict[str, Any], samples: List[Dict[str, Any]], 
                            dataset_name: str) -> None:
        """Add benchmark details to pipeline info if dataset is BenchHub."""
        if dataset_name != "benchhub" or not samples:
            return

        unique_benchmark_names = set()
        for sample in samples:
            if "metadata" in sample and "benchmark_name" in sample["metadata"]:
                unique_benchmark_names.add(sample["metadata"]["benchmark_name"])
        
        if not unique_benchmark_names:
            return

        benchmark_details = {}
        for bn_name in unique_benchmark_names:
            for entry in BENCHHUB_INFO_ENTRIES:
                if entry.dataset == bn_name:
                    benchmark_details[bn_name] = {
                        "citation_key": entry.citation_key,
                        "citation": entry.citation,
                        "license": entry.license,
                        "anthology": entry.anthology,
                        "languages": entry.languages,
                    }
                    break
        
        if benchmark_details:
            pipeline_info['benchmark_details'] = benchmark_details
            logger.info(f"Added citation information for BenchHub benchmarks: {list(benchmark_details.keys())}")


class InferenceEngine:
    """Handles model inference and scaling operations."""
    
    def __init__(self, components: ComponentManager):
        self.components = components

    def run_inference(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run inference on the provided samples."""
        try:
            if self.components.scaler:
                logger.info(f"Applying scaling method: {self.components.config.scaling_method_name} "
                           f"to {len(samples)} samples.")
                predictions = self.components.scaler.apply(samples)
            else:
                logger.info(f"Performing direct model inference for {len(samples)} samples.")
                predictions = self.components.model.generate_batch(samples)
            
            logger.info(f"Inference completed for {len(predictions)} items.")
            return predictions
            
        except Exception as e:
            logger.error(f"Error during model inference or scaling: {e}", exc_info=True)
            raise


class PipelineRunner:
    """
    Refactored PipelineRunner that orchestrates the entire LLM evaluation pipeline.
    
    This class now uses composition with specialized handler classes to manage
    different aspects of the pipeline, making it more maintainable and testable.
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
        language_penalize: bool = True,
        target_lang: str = "ko",
        custom_cot_parser: Optional[Callable[[str], Tuple[str, str]]] = None,
        num_few_shot: Optional[int] = 0,
        few_shot_split: Optional[str] = None,
        few_shot_instruction: Optional[str] = DEFAULT_FEW_SHOT_INSTRUCTION,
        few_shot_example_template: str = DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE,
    ):
        """
        Initialize the PipelineRunner with configuration parameters.

        Args:
            dataset_name: Dataset identifier (registered in the dataset registry).
            subset: Optional sub-task or configuration (e.g., "csat_geo").
            split: Dataset split to load ("train", "valid", "test", etc.).
            model_backend_name: Model backend identifier (e.g., "huggingface", "openai", "multi").
            scaling_method_name: Scaling (decoding) method identifier.
            evaluation_method_name: Evaluator identifier.
            dataset_params: Additional parameters for the dataset loader.
            model_backend_params: Additional parameters for the model backend.
            scaling_params: Parameters for the scaling method.
            evaluator_params: Parameters for the evaluator.
            language_penalize: If True, apply language penalizer to predictions.
            target_lang: Target language code for penalization.
            custom_cot_parser: Optional custom chain-of-thought parser function.
            num_few_shot: Number of few-shot examples to use.
            few_shot_split: Split to use for few-shot examples.
            few_shot_instruction: Instruction template for few-shot examples.
            few_shot_example_template: Example template for few-shot examples.
        """
        # Create configuration object
        self.config = PipelineConfig(
            dataset_name=dataset_name,
            subset=subset,
            split=split,
            model_backend_name=model_backend_name,
            scaling_method_name=scaling_method_name,
            evaluation_method_name=evaluation_method_name,
            dataset_params=dataset_params,
            model_backend_params=model_backend_params,
            scaling_params=scaling_params,
            evaluator_params=evaluator_params,
            language_penalize=language_penalize,
            target_lang=target_lang,
            custom_cot_parser=custom_cot_parser,
            num_few_shot=num_few_shot if num_few_shot is not None else 0,
            few_shot_split=few_shot_split,
            few_shot_instruction=few_shot_instruction,
            few_shot_example_template=few_shot_example_template,
        )

        # Initialize specialized handlers
        self.components = ComponentManager(self.config)
        self.few_shot_handler = FewShotHandler(self.config)
        self.language_penalizer = LanguagePenalizer(self.config)
        self.inference_engine = InferenceEngine(self.components)

        # Load all components
        self.components.load_all_components()

    @property
    def dataset(self) -> Optional[BaseDataset]:
        """Access to the loaded dataset."""
        return self.components.dataset

    @property
    def model(self) -> Optional[BaseModel]:
        """Access to the loaded model."""
        return self.components.model

    @property
    def scaler(self) -> Optional[BaseScalingMethod]:
        """Access to the loaded scaler."""
        return self.components.scaler

    @property
    def evaluator(self) -> Optional[BaseEvaluator]:
        """Access to the loaded evaluator."""
        return self.components.evaluator
        
    def run(self) -> EvaluationResult:
        """
        Execute the entire evaluation pipeline.

        Returns:
            EvaluationResult: An object encapsulating final metrics, samples, and pipeline info.
        """
        self._validate_components()
        
        start_time = time.time()
        logger.info(f"Pipeline run started for dataset: {self.config.dataset_name}, "
                   f"split: {self.config.split}, model: {self.config.model_backend_name}")

        try:
            # Step 1: Load and prepare data
            raw_samples = self._load_evaluation_data()
            few_shot_prefix = self.few_shot_handler.prepare_few_shot_prefix()
            evaluation_samples = self.few_shot_handler.filter_evaluation_samples(raw_samples)
            
            if not evaluation_samples:
                return self._create_empty_result(few_shot_prefix, "No samples for evaluation after few-shot processing")

            # Step 2: Prepare samples for inference
            inference_samples = self.few_shot_handler.process_samples_for_inference(
                evaluation_samples, few_shot_prefix
            )
            logger.info(f"Prepared {len(inference_samples)} samples for model inference.")

            # Step 3: Run inference
            predictions = self.inference_engine.run_inference(inference_samples)

            # Step 4: Evaluate predictions
            eval_dict = self._run_evaluation(predictions)

            # Step 5: Apply language penalization if enabled
            self.language_penalizer.apply_penalization(eval_dict)

            # Step 6: Create final result
            return self._create_final_result(eval_dict, few_shot_prefix, start_time)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return self._create_error_result(str(e))

    def _validate_components(self) -> None:
        """Validate that all required components are loaded."""
        if not all([self.components.dataset, self.components.model, self.components.evaluator]):
            raise RuntimeError("Pipeline components are not fully loaded.")

    def _load_evaluation_data(self) -> List[Dict[str, Any]]:
        """Load the main evaluation data from the dataset."""
        raw_samples = self.components.dataset.load()
        logger.info(f"Loaded {len(raw_samples)} samples for evaluation from split '{self.config.split}'.")
        return raw_samples

    def _run_evaluation(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation on the predictions."""
        logger.info(f"Starting evaluation with '{self.config.evaluation_method_name}'.")
        try:
            return self.components.evaluator.evaluate(predictions, model=self.components.model)
        except Exception as e:
            logger.error(f"Error during evaluation with '{self.config.evaluation_method_name}': {e}", exc_info=True)
            raise

    def _create_pipeline_info(self, few_shot_prefix: str, elapsed_time: float) -> Dict[str, Any]:
        """Create pipeline information dictionary."""
        pipeline_info = {
            "dataset_name": self.config.dataset_name,
            "subset": self.config.subset,
            "split": self.config.split,
            "model_backend_name": self.config.model_backend_name,
            "scaling_method_name": self.config.scaling_method_name,
            "evaluation_method_name": self.config.evaluation_method_name,
            "elapsed_time_sec": elapsed_time,
            "num_few_shot_configured": self.config.num_few_shot,
            "num_few_shot_applied": self.config.num_few_shot if few_shot_prefix else 0,
            "few_shot_source_split": self.config.few_shot_split if self.config.num_few_shot > 0 and few_shot_prefix else None,
            "language_penalize_enabled": self.config.language_penalize,
            "target_lang_for_penalty": self.config.target_lang if self.config.language_penalize else None,
        }
        return pipeline_info

    def _create_final_result(self, eval_dict: Dict[str, Any], few_shot_prefix: str, 
                           start_time: float) -> EvaluationResult:
        """Create the final evaluation result."""
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline run completed in {elapsed_time:.2f} seconds.")

        pipeline_info = self._create_pipeline_info(few_shot_prefix, elapsed_time)
        
        # Add BenchHub details if applicable
        final_samples = eval_dict.get("samples", [])
        BenchHubInfoProcessor.add_benchmark_details(pipeline_info, final_samples, self.config.dataset_name)

        # Merge with existing info from evaluator
        existing_info = eval_dict.get("info", {})
        merged_info = {**existing_info, **pipeline_info}

        return EvaluationResult(
            metrics=eval_dict.get("metrics", {}),
            samples=final_samples,
            info=merged_info
        )

    def _create_empty_result(self, few_shot_prefix: str, error_msg: str) -> EvaluationResult:
        """Create an empty result for cases with no evaluation samples."""
        logger.warning(f"{error_msg}. Returning empty result.")
        return EvaluationResult(
            metrics={}, 
            samples=[], 
            info={
                "dataset_name": self.config.dataset_name,
                "subset": self.config.subset,
                "split": self.config.split,
                "model_backend_name": self.config.model_backend_name,
                "num_few_shot_applied": self.config.num_few_shot if few_shot_prefix else 0,
                "error": error_msg
            }
        )

    def _create_error_result(self, error_msg: str) -> EvaluationResult:
        """Create an error result for pipeline failures."""
        return EvaluationResult(
            metrics={"pipeline_error": error_msg}, 
            samples=[], 
            info={
                "dataset_name": self.config.dataset_name,
                "subset": self.config.subset,
                "split": self.config.split,
                "model_backend_name": self.config.model_backend_name,
                "error": f"Pipeline failed: {error_msg}"
            }
        )