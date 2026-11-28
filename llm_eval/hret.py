#!/usr/bin/env python
"""
HRET (Haerae Evaluation Toolkit) - MLOps-friendly Facade API

This module provides a deepeval-style decorator-based API for easy integration
with MLOps pipelines and other programs. It offers a high-level interface
for LLM evaluation with minimal boilerplate code.

Example usage:
    ```python
    import hret
    
    @hret.evaluate(dataset="kmmlu", model="gpt-4")
    def my_model_function(input_text):
        return model.generate(input_text)
    
    # Or use context manager
    with hret.evaluation_context(dataset="kmmlu") as ctx:
        results = ctx.evaluate(my_function)
        ctx.log_to_mlflow()
    ```
"""

import functools
import logging
import os
import time
import json
import yaml
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Tuple
from collections import defaultdict

from llm_eval.models import BaseModel, register_model
from llm_eval.runner import PipelineRunner, PipelineConfig
from llm_eval.utils.util import EvaluationResult
from llm_eval.utils.logging import get_logger

logger = get_logger(name="hret", level=logging.INFO)


@dataclass
class HRETConfig:
    """Global configuration for HRET."""
    # Default evaluation settings
    default_dataset: str = "kmmlu"
    default_model: str = "huggingface"
    default_split: str = "test"
    default_evaluation_method: str = "string_match"
    
    # MLOps integration settings
    mlflow_tracking: bool = False
    wandb_tracking: bool = False
    tensorboard_tracking: bool = False
    
    # Logging and output settings
    log_level: str = "INFO"
    output_dir: str = "./hret_results"
    auto_save_results: bool = True
    
    # Performance settings
    batch_size: Optional[int] = None
    max_workers: Optional[int] = None
    
    # Advanced settings
    custom_loggers: List[str] = field(default_factory=list)
    config_file: Optional[str] = None
    
    def __post_init__(self):
        """Load configuration from file if specified."""
        if self.config_file and Path(self.config_file).exists():
            self.load_from_file(self.config_file)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configuration
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")


# Global configuration instance
_global_config = HRETConfig()


def configure(**kwargs) -> None:
    """Configure global HRET settings."""
    global _global_config
    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)
        else:
            logger.warning(f"Unknown configuration key: {key}")


def load_config(config_path: str) -> None:
    """Load configuration from file."""
    global _global_config
    _global_config.load_from_file(config_path)


class MetricsTracker:
    """Tracks and manages evaluation metrics across multiple runs."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_run_metrics: Dict[str, Any] = {}
        self.run_metadata: Dict[str, Any] = {}
    
    def start_run(self, run_name: Optional[str] = None, **metadata) -> None:
        """Start a new evaluation run."""
        self.current_run_metrics = {}
        self.run_metadata = {
            "run_name": run_name or f"run_{int(time.time())}",
            "start_time": time.time(),
            **metadata
        }
        logger.info(f"Started evaluation run: {self.run_metadata['run_name']}")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for the current run."""
        self.current_run_metrics.update(metrics)
    
    def end_run(self) -> Dict[str, Any]:
        """End the current run and return results."""
        self.run_metadata["end_time"] = time.time()
        self.run_metadata["duration"] = self.run_metadata["end_time"] - self.run_metadata["start_time"]
        
        run_result = {
            "metadata": self.run_metadata.copy(),
            "metrics": self.current_run_metrics.copy()
        }
        
        self.metrics_history.append(run_result)
        logger.info(f"Completed evaluation run: {self.run_metadata['run_name']}")
        
        return run_result
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get all metrics history."""
        return self.metrics_history.copy()
    
    def compare_runs(self, metric_name: str) -> Dict[str, Any]:
        """Compare a specific metric across all runs."""
        comparison = {}
        for run in self.metrics_history:
            run_name = run["metadata"]["run_name"]
            if metric_name in run["metrics"]:
                comparison[run_name] = run["metrics"][metric_name]
        
        if comparison:
            values = list(comparison.values())
            comparison["_stats"] = {
                "best": max(values) if isinstance(values[0], (int, float)) else None,
                "worst": min(values) if isinstance(values[0], (int, float)) else None,
                "average": sum(values) / len(values) if all(isinstance(v, (int, float)) for v in values) else None
            }
        
        return comparison


@register_model("callable")
class FunctionModel(BaseModel):
    """Run an in-process prediction function once per sample."""

    def __init__(self, function: Callable[[str], str], **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def generate_batch(self, inputs, return_logits=False, **kwargs):
        if return_logits:
            raise ValueError("A text prediction function cannot supply token log probabilities.")
        results = []
        for item in inputs:
            try:
                results.append({**item, "prediction": self.function(item["input"])})
            except Exception as exc:
                results.append({**item, "prediction": "", "error": str(exc)})
        return results


class EvaluationContext:
    """Context manager for evaluation sessions."""
    
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        model_backend_name: Optional[str] = None,
        run_name: Optional[str] = None,
        **kwargs
    ):
        self.config = PipelineConfig(
            dataset_name=dataset_name or _global_config.default_dataset,
            model_backend_name=model_backend_name or _global_config.default_model,
            evaluation_method_name=kwargs.get('evaluation_method_name') or _global_config.default_evaluation_method,
            split=kwargs.get('split', _global_config.default_split),
            **{k: v for k, v in kwargs.items() if k not in ['evaluation_method_name', 'split']}
        )
        
        self.metrics_tracker = MetricsTracker()
        self.run_name = run_name
        self.results: List[EvaluationResult] = []
        self.mlops_integrations: List[Callable] = []
        
    def __enter__(self) -> 'EvaluationContext':
        """Enter the evaluation context."""
        self.metrics_tracker.start_run(
            run_name=self.run_name,
            dataset=self.config.dataset_name,
            model=self.config.model_backend_name,
            config=self.config.tracking_metadata()
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the evaluation context."""
        run_result = self.metrics_tracker.end_run()
        
        # Execute MLOps integrations
        for integration in self.mlops_integrations:
            try:
                integration(run_result, self.results)
            except Exception as e:
                logger.error(f"MLOps integration failed: {e}")
        
        # Auto-save results if enabled
        if _global_config.auto_save_results:
            self.save_results()
    
    def evaluate(self, model_function: Callable[[str], str]) -> EvaluationResult:
        """Evaluate a model function."""
        config = vars(self.config).copy()
        generator_params = {"function": model_function}
        if self.config.model_backend_name == "multi":
            config["model_backend_params"] = {
                **self.config.model_backend_params,
                "generate_model": {"name": "callable", "params": generator_params},
            }
        else:
            config.update(model_backend_name="callable", model_backend_params=generator_params)
        runner = PipelineRunner(**config)
        result = runner.run()
        self.results.append(result)
        self.metrics_tracker.log_metrics(result.metrics)
        
        return result
    
    def benchmark(self, model_functions: Dict[str, Callable[[str], str]]) -> Dict[str, EvaluationResult]:
        """Benchmark multiple model functions."""
        results = {}
        for name, func in model_functions.items():
            logger.info(f"Benchmarking model: {name}")
            result = self.evaluate(func)
            results[name] = result
        
        return results
    
    def add_mlops_integration(self, integration_func: Callable) -> None:
        """Add an MLOps integration function."""
        self.mlops_integrations.append(integration_func)
    
    def log_to_mlflow(self, experiment_name: Optional[str] = None) -> None:
        """Log results to MLflow."""
        try:
            import mlflow
            
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            def mlflow_logger(run_result, results):
                with mlflow.start_run(run_name=run_result["metadata"]["run_name"]):
                    # Log metrics
                    for key, value in run_result["metrics"].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                    
                    # Log parameters
                    for key, value in run_result["metadata"]["config"].items():
                        if isinstance(value, (str, int, float, bool)):
                            mlflow.log_param(key, value)
                    
                    # Log artifacts
                    if results:
                        for i, result in enumerate(results):
                            artifact_path = f"result_{i}.json"
                            with open(artifact_path, 'w') as f:
                                json.dump(result.to_dict(), f, indent=2)
                            mlflow.log_artifact(artifact_path)
                            os.remove(artifact_path)
            
            self.add_mlops_integration(mlflow_logger)
            logger.info("MLflow integration added")
            
        except ImportError:
            logger.error("MLflow not installed. Install with: pip install mlflow")
    
    def log_to_wandb(self, project_name: Optional[str] = None) -> None:
        """Log results to Weights & Biases."""
        try:
            import wandb
            
            def wandb_logger(run_result, results):
                wandb.init(
                    project=project_name or "hret-evaluation",
                    name=run_result["metadata"]["run_name"],
                    config=run_result["metadata"]["config"]
                )
                
                # Log metrics
                wandb.log(run_result["metrics"])
                
                # Log results as artifacts
                for i, result in enumerate(results):
                    artifact = wandb.Artifact(f"evaluation_result_{i}", type="result")
                    with artifact.new_file(f"result_{i}.json") as f:
                        json.dump(result.to_dict(), f, indent=2)
                    wandb.log_artifact(artifact)
                
                wandb.finish()
            
            self.add_mlops_integration(wandb_logger)
            logger.info("Weights & Biases integration added")
            
        except ImportError:
            logger.error("wandb not installed. Install with: pip install wandb")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save evaluation results to file."""
        if not filename:
            timestamp = int(time.time())
            filename = f"hret_results_{timestamp}.json"
        
        filepath = Path(_global_config.output_dir) / filename
        
        output_data = {
            "run_metadata": self.metrics_tracker.run_metadata,
            "metrics": self.metrics_tracker.current_run_metrics,
            "results": [result.to_dict() for result in self.results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)


# Decorator functions
def evaluate(
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    evaluation_method: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Decorator to evaluate a model function.
    
    Args:
        dataset: Dataset name to use for evaluation
        model: Model backend name
        evaluation_method: Evaluation method to use
        **kwargs: Additional configuration parameters
    
    Example:
        ```python
        @hret.evaluate(dataset="kmmlu", model="huggingface")
        def my_model(input_text):
            return model.generate(input_text)
        
        result = my_model()  # Returns EvaluationResult
        ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **wrapper_kwargs) -> EvaluationResult:
            with evaluation_context(
                dataset_name=dataset,
                model_backend_name=model,
                evaluation_method_name=evaluation_method,
                run_name=f"eval_{func.__name__}",
                **kwargs
            ) as ctx:
                return ctx.evaluate(func)
        
        return wrapper
    return decorator


def benchmark(
    dataset: Optional[str] = None,
    models: Optional[List[str]] = None,
    **kwargs
) -> Callable:
    """
    Decorator to benchmark multiple model functions.
    
    Args:
        dataset: Dataset name to use for benchmarking
        models: List of model backend names to compare
        **kwargs: Additional configuration parameters
    
    Example:
        ```python
        @hret.benchmark(dataset="kmmlu", models=["gpt-4", "claude-3"])
        def model_comparison():
            return {
                "gpt-4": lambda x: gpt4_model.generate(x),
                "claude-3": lambda x: claude_model.generate(x)
            }
        
        results = model_comparison()  # Returns Dict[str, EvaluationResult]
        ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **wrapper_kwargs) -> Dict[str, EvaluationResult]:
            model_functions = func(*args, **wrapper_kwargs)
            
            with evaluation_context(
                dataset_name=dataset,
                run_name=f"benchmark_{func.__name__}",
                **kwargs
            ) as ctx:
                return ctx.benchmark(model_functions)
        
        return wrapper
    return decorator


def track_metrics(metrics: List[str]) -> Callable:
    """
    Decorator to track specific metrics from a function.
    
    Args:
        metrics: List of metric names to track
    
    Example:
        ```python
        @hret.track_metrics(["accuracy", "f1_score"])
        def my_evaluation():
            # Your evaluation logic
            return {"accuracy": 0.85, "f1_score": 0.82}
        ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                tracked_metrics = {k: v for k, v in result.items() if k in metrics}
                logger.info(f"Tracked metrics: {tracked_metrics}")
            
            return result
        
        return wrapper
    return decorator


# Context manager function
@contextmanager
def evaluation_context(
    dataset_name: Optional[str] = None,
    model_backend_name: Optional[str] = None,
    run_name: Optional[str] = None,
    **kwargs
) -> Iterator[EvaluationContext]:
    """
    Context manager for evaluation sessions.
    
    Example:
        ```python
        with hret.evaluation_context(dataset="kmmlu") as ctx:
            result = ctx.evaluate(my_model_function)
            ctx.log_to_mlflow()
        ```
    """
    ctx = EvaluationContext(
        dataset_name=dataset_name,
        model_backend_name=model_backend_name,
        run_name=run_name,
        **kwargs
    )
    
    try:
        yield ctx.__enter__()
    finally:
        ctx.__exit__(None, None, None)


# Convenience functions
def quick_eval(
    model_function: Callable[[str], str],
    dataset: str = "kmmlu",
    **kwargs
) -> EvaluationResult:
    """
    Quick evaluation function for simple use cases.
    
    Args:
        model_function: Function that takes input text and returns prediction
        dataset: Dataset name to use
        **kwargs: Additional configuration
    
    Returns:
        EvaluationResult: Evaluation results
    """
    with evaluation_context(dataset_name=dataset, **kwargs) as ctx:
        return ctx.evaluate(model_function)


def compare_models(
    model_functions: Dict[str, Callable[[str], str]],
    dataset: str = "kmmlu",
    **kwargs
) -> Dict[str, EvaluationResult]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        model_functions: Dictionary mapping model names to functions
        dataset: Dataset name to use
        **kwargs: Additional configuration
    
    Returns:
        Dict[str, EvaluationResult]: Results for each model
    """
    with evaluation_context(dataset_name=dataset, **kwargs) as ctx:
        return ctx.benchmark(model_functions)


# Global metrics tracker
_global_tracker = MetricsTracker()


def get_metrics_history() -> List[Dict[str, Any]]:
    """Get global metrics history."""
    return _global_tracker.get_history()


def compare_metric(metric_name: str) -> Dict[str, Any]:
    """Compare a metric across all runs."""
    return _global_tracker.compare_runs(metric_name)


# Export main API
__all__ = [
    # Configuration
    'configure',
    'load_config',
    'HRETConfig',
    
    # Decorators
    'evaluate',
    'benchmark',
    'track_metrics',
    
    # Context managers
    'evaluation_context',
    'EvaluationContext',
    
    # Convenience functions
    'quick_eval',
    'compare_models',
    
    # Metrics tracking
    'MetricsTracker',
    'get_metrics_history',
    'compare_metric',
]