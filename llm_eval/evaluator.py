"""
Evaluator CLI

This module provides a high-level interface to run the full LLM evaluation pipeline,
which includes dataset loading, model inference (with optional scaling),
evaluation, and additional post-processing such as language penalization.
It also supports few-shot prompting by preparing examples and prepending them
to the main task prompt within the PipelineRunner.
"""

import argparse
import json
import logging  # Standard logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from llm_eval.runner import PipelineRunner
from llm_eval.utils.logging import \
    get_logger  # Using the project's logger setup
from llm_eval.utils.prompt_template import (DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE,
                                            DEFAULT_FEW_SHOT_INSTRUCTION)
from llm_eval.utils.util import (  # _load_function for dynamic imports
    EvaluationResult, _load_function)

# Logger for this Evaluator.py module
logger = get_logger(name=__name__, level=logging.INFO)

def _parse_json_str(json_str: Optional[str]) -> Dict[str, Any]:
    """
    Parses a JSON string into a dictionary.
    If the input string is None or empty, returns an empty dictionary.
    If parsing fails due to a JSONDecodeError, logs a warning and returns {}.
    """
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON parameter at line %d, column %d. Using defaults.", e.lineno, e.colno)
        return {}

class Evaluator:
    """
    High-level interface for running the full LLM evaluation pipeline.
    This class leverages PipelineRunner internally to connect dataset, model,
    scaling method, and evaluator components. It supports judge/reward backends
    via MultiModel if specified.
    It also handles few-shot prompting by passing relevant parameters to PipelineRunner.
    """

    def __init__(
        self,
        default_model_backend: str = "huggingface",
        default_judge_backend: Optional[str] = None,
        default_reward_backend: Optional[str] = None,
        default_scaling_method: Optional[str] = None,
        default_evaluation_method: str = "string_match",
        default_split: str = "test",
        default_cot_parser: Optional[Union[Callable[[str], Tuple[str, str]], str]] = None,
        # --- New default few-shot parameters ---
        default_num_few_shot: int = 0,
        default_few_shot_split: Optional[str] = None,
        default_few_shot_instruction: str = DEFAULT_FEW_SHOT_INSTRUCTION,
        default_few_shot_example_template: str = DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE,
        # ---
    ):
        """
        Initializes the Evaluator with default backend configurations and few-shot settings.

        Args:
            default_model_backend (str): Default generation model backend.
            default_judge_backend (Optional[str]): Default judge model backend (if any).
            default_reward_backend (Optional[str]): Default reward model backend (if any).
            default_scaling_method (Optional[str]): Default scaling method (e.g., "beam_search").
            default_evaluation_method (str): Default evaluation method (e.g., "string_match").
            default_split (str): Default dataset split (e.g., "test").
            default_cot_parser (callable or str, optional): Default custom chain-of-thought parser.
                Can be provided as a callable or as a full function path string.
            default_num_few_shot (int): Default number of few-shot examples.
            default_few_shot_split (Optional[str]): Default split to source few-shot examples from.
            default_few_shot_instruction (str): Default instruction for few-shot examples.
            default_few_shot_example_template (str): Default template for formatting each few-shot example.
        """
        self.default_model_backend = default_model_backend
        self.default_judge_backend = default_judge_backend
        self.default_reward_backend = default_reward_backend
        self.default_scaling_method = default_scaling_method
        self.default_evaluation_method = default_evaluation_method
        self.default_split = default_split

        # Process default_cot_parser:
        if isinstance(default_cot_parser, str):
            try:
                self.default_cot_parser = _load_function(default_cot_parser)
                logger.info(f"Default CoT parser loaded from: {default_cot_parser}")
            except Exception as e:
                logger.error(f"Failed to load default CoT parser from '{default_cot_parser}': {e}. No default CoT parser will be set.", exc_info=True)
                self.default_cot_parser = None
        else:
            self.default_cot_parser = default_cot_parser # Assign if already a callable or None

        # Store default few-shot params
        self.default_num_few_shot = default_num_few_shot
        self.default_few_shot_split = default_few_shot_split
        self.default_few_shot_instruction = default_few_shot_instruction
        self.default_few_shot_example_template = default_few_shot_example_template
        logger.debug(f"Evaluator initialized with default few-shot params: num={self.default_num_few_shot}, split='{self.default_few_shot_split}'")

    def run(
        self,
        model: Optional[str] = None,
        judge_model: Optional[str] = None,
        reward_model: Optional[str] = None,
        dataset: str = "haerae_bench",
        subset: Union[str, List[str], None] = None,
        split: Optional[str] = None,
        scaling_method: Optional[str] = None,
        evaluation_method: Optional[str] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        judge_params: Optional[Dict[str, Any]] = None,
        reward_params: Optional[Dict[str, Any]] = None,
        scaling_params: Optional[Dict[str, Any]] = None,
        evaluator_params: Optional[Dict[str, Any]] = None,
        language_penalize: bool = True,
        target_lang: Optional[str] = "ko",
        custom_cot_parser: Optional[Union[Callable[[str], Tuple[str, str]], str]] = None,
        # --- New few-shot parameters for run method ---
        num_few_shot: Optional[int] = None,
        few_shot_split: Optional[str] = None,
        few_shot_instruction: Optional[str] = None,
        few_shot_example_template: Optional[str] = None,
        # ---
    ) -> EvaluationResult:
        """
        Runs the full LLM evaluation pipeline. This involves:
          1. Determining final configurations for all components (model, dataset, evaluators, etc.).
          2. Instantiating PipelineRunner with these configurations.
          3. Executing the pipeline via PipelineRunner.run().
          4. Returning the evaluation results.

        Args:
            model (Optional[str]): Main model backend name. Defaults to default_model_backend.
            judge_model (Optional[str]): Judge model backend name.
            reward_model (Optional[str]): Reward model backend name.
            dataset (str): Dataset name (registry key).
            subset (Union[str, List[str], None]): Dataset subset/config name.
            split (Optional[str]): Dataset split (e.g., "train", "test"). Defaults to default_split.
            scaling_method (Optional[str]): Scaling method name (registry key).
            evaluation_method (Optional[str]): Evaluation method name (registry key).
            dataset_params (Optional[Dict[str, Any]]): Additional parameters for dataset loading.
            model_params (Optional[Dict[str, Any]]): Additional parameters for the main model.
            judge_params (Optional[Dict[str, Any]]): Additional parameters for the judge model.
            reward_params (Optional[Dict[str, Any]]): Additional parameters for the reward model.
            scaling_params (Optional[Dict[str, Any]]): Additional parameters for the scaling method.
            evaluator_params (Optional[Dict[str, Any]]): Additional parameters for the evaluator.
            language_penalize (bool): If True, apply the language penalizer to predictions.
            target_lang (Optional[str]): Target language for penalizer. Defaults to "ko".
            custom_cot_parser (callable or str, optional): A custom CoT parser function or its Python path.
                                                       Overrides default if provided.
            num_few_shot (Optional[int]): Number of few-shot examples. Overrides default.
            few_shot_split (Optional[str]): Split for sourcing few-shot examples. Overrides default.
            few_shot_instruction (Optional[str]): Instruction for few-shot block. Overrides default.
            few_shot_example_template (Optional[str]): Template for each few-shot example. Overrides default.

        Returns:
            EvaluationResult: Object containing evaluation metrics, sample outputs, and run info.
        """

        logger.info(f"Evaluator.run initiated. Dataset: '{dataset}', Model: '{model or self.default_model_backend}', Eval Method: '{evaluation_method or self.default_evaluation_method}'")

        # Determine final values for parameters, prioritizing run() arguments over defaults
        model_backend_name = model or self.default_model_backend
        judge_backend_name = judge_model or self.default_judge_backend
        reward_backend_name = reward_model or self.default_reward_backend

        final_split = split if split is not None else self.default_split
        final_scaling = scaling_method if scaling_method is not None else self.default_scaling_method
        final_eval_method = evaluation_method if evaluation_method is not None else self.default_evaluation_method

        # Determine custom_cot_parser: use argument if provided, else default.
        # PipelineRunner will handle loading if it's a string path.
        final_custom_cot_parser = custom_cot_parser
        if final_custom_cot_parser is None:
            final_custom_cot_parser = self.default_cot_parser

        # Determine final few-shot parameters
        final_num_few_shot = num_few_shot if num_few_shot is not None else self.default_num_few_shot
        final_few_shot_split = few_shot_split if few_shot_split is not None else self.default_few_shot_split
        final_few_shot_instruction = few_shot_instruction if few_shot_instruction is not None else self.default_few_shot_instruction
        final_few_shot_example_template = few_shot_example_template if few_shot_example_template is not None else self.default_few_shot_example_template

        logger.debug(f"Final few-shot params for PipelineRunner: num={final_num_few_shot}, split='{final_few_shot_split}'")

        current_model_params = model_params or {}
        # Note: custom_cot_parser is passed directly to PipelineRunner now, not via model_params here.

        # Determine if MultiModel is needed based on whether judge or reward models are specified.
        use_multi = (judge_backend_name is not None) or (reward_backend_name is not None)
        if use_multi:
            logger.info("Configuring MultiModel due to specified judge_model or reward_model.")
            multi_config = {
                "generate_model": {"name": model_backend_name, "params": current_model_params.copy()} if model_backend_name else None,
                "judge_model": {"name": judge_backend_name, "params": judge_params or {}} if judge_backend_name else None,
                "reward_model": {"name": reward_backend_name, "params": reward_params or {}} if reward_backend_name else None,
            }
            runner_model_backend_name = "multi"
            runner_model_backend_params = multi_config
        else:
            runner_model_backend_name = model_backend_name
            runner_model_backend_params = current_model_params

        try:
            runner = PipelineRunner(
                dataset_name=dataset,
                subset=subset,
                split=final_split,
                model_backend_name=runner_model_backend_name,
                model_backend_params=runner_model_backend_params,
                scaling_method_name=final_scaling,
                scaling_params=scaling_params or {},
                evaluation_method_name=final_eval_method,
                evaluator_params=evaluator_params or {},
                dataset_params=dataset_params or {},
                language_penalize=language_penalize,
                target_lang=target_lang if target_lang is not None else "ko",
                custom_cot_parser=final_custom_cot_parser, # Pass determined CoT parser to PipelineRunner
                num_few_shot=final_num_few_shot,
                few_shot_split=final_few_shot_split,
                few_shot_instruction=final_few_shot_instruction,
                few_shot_example_template=final_few_shot_example_template,
            )
            logger.info("PipelineRunner instantiated. Starting pipeline execution...")
            eval_result = runner.run()
            logger.info(f"Evaluation successfully completed for dataset '{dataset}'. Metrics: {eval_result.metrics if eval_result else 'N/A'}")
            return eval_result
        except Exception as e:
            logger.critical(f"PipelineRunner execution failed for dataset '{dataset}': {e}", exc_info=True)
            # Return a meaningful EvaluationResult indicating failure
            return EvaluationResult(
                metrics={"pipeline_error": str(e)},
                samples=[],
                info={
                    "dataset_name": dataset, "subset": subset, "split": final_split,
                    "model_backend_name": runner_model_backend_name, "error_message": str(e),
                    "status": "failed"
                }
            )


def run_from_config(config_path: str) -> EvaluationResult:
    """Run the evaluation pipeline using a YAML/JSON configuration file.

    The configuration file should define all components required for the
    :class:`Evaluator` pipeline (dataset, model, evaluation method, etc.).
    See ``examples/evaluator_config.yaml`` for a full template.

    Args:
        config_path: Path to a YAML or JSON file containing evaluator settings.

    Returns:
        EvaluationResult: The result of the evaluation run.
    """

    logger.info(f"Loading evaluator configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    judge_cfg = config.get("judge_model", {})
    reward_cfg = config.get("reward_model", {})
    scaling_cfg = config.get("scaling", {})
    eval_cfg = config.get("evaluation", {})
    few_shot_cfg = config.get("few_shot", {})

    evaluator = Evaluator(
        default_model_backend=model_cfg.get("name", "huggingface"),
        default_judge_backend=judge_cfg.get("name"),
        default_reward_backend=reward_cfg.get("name"),
        default_scaling_method=scaling_cfg.get("name"),
        default_evaluation_method=eval_cfg.get("method", "string_match"),
        default_split=dataset_cfg.get("split", "test"),
        default_cot_parser=config.get("custom_cot_parser"),
        default_num_few_shot=few_shot_cfg.get("num", 0),
        default_few_shot_split=few_shot_cfg.get("split"),
        default_few_shot_instruction=few_shot_cfg.get(
            "instruction", DEFAULT_FEW_SHOT_INSTRUCTION
        ),
        default_few_shot_example_template=few_shot_cfg.get(
            "example_template", DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE
        ),
    )

    return evaluator.run(
        model=model_cfg.get("name"),
        judge_model=judge_cfg.get("name"),
        reward_model=reward_cfg.get("name"),
        dataset=dataset_cfg.get("name", "haerae_bench"),
        subset=dataset_cfg.get("subset"),
        split=dataset_cfg.get("split"),
        scaling_method=scaling_cfg.get("name"),
        evaluation_method=eval_cfg.get("method"),
        dataset_params=dataset_cfg.get("params"),
        model_params=model_cfg.get("params"),
        judge_params=judge_cfg.get("params"),
        reward_params=reward_cfg.get("params"),
        scaling_params=scaling_cfg.get("params"),
        evaluator_params=eval_cfg.get("params"),
        language_penalize=config.get("language_penalize", True),
        target_lang=config.get("target_lang", "ko"),
        custom_cot_parser=config.get("custom_cot_parser"),
        num_few_shot=few_shot_cfg.get("num"),
        few_shot_split=few_shot_cfg.get("split"),
        few_shot_instruction=few_shot_cfg.get("instruction"),
        few_shot_example_template=few_shot_cfg.get("example_template"),
    )

def main():
    """
    Command-line entry point for the Evaluator CLI.
    Supports configuration via arguments for dataset, model, scaling, evaluation,
    few-shot prompting, and other parameters.
    The final evaluation results are output as JSON.
    """
    parser = argparse.ArgumentParser(
        description="LLM Evaluator CLI. Supports Few-Shot, Judge/Reward models, and Custom CoT Parsers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # --- Required arguments ---
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (registry key). (e.g., haerae_bench)")
    parser.add_argument("--model", type=str, required=True, help="Main model backend name. (e.g., huggingface)")

    # --- Optional arguments with defaults ---
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config name. Use comma-separated values for multiple subsets if supported by the dataset loader. (e.g., csat_geo or csat_geo,csat_law)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on.")
    parser.add_argument("--evaluation_method", type=str, default="string_match", help="Evaluation method (registry key).")

    parser.add_argument("--judge_model", type=str, default=None, help="Judge model backend name (optional).")
    parser.add_argument("--reward_model", type=str, default=None, help="Reward model backend name (optional).")
    parser.add_argument("--scaling_method", type=str, default=None, help="Scaling method (registry key, optional).")

    parser.add_argument("--output_file", type=str, default=None, help="File path to save JSON results. Prints to stdout if not set.")
    parser.add_argument("--cot_parser", type=str, default=None, help="Full Python path to a custom CoT parser function (e.g., 'my_package.my_module.my_parser'). This will be used as the default if not overridden in model_params.")

    # JSON string parameters for finer control
    parser.add_argument("--dataset_params", type=str, default="{}", help="JSON string for dataset-specific parameters.")
    parser.add_argument("--model_params", type=str, default="{}", help="JSON string for main model parameters. Can include 'cot_parser' to override the default or CLI --cot_parser.")
    parser.add_argument("--judge_params", type=str, default="{}", help="JSON string for judge model parameters.")
    parser.add_argument("--reward_params", type=str, default="{}", help="JSON string for reward model parameters.")
    parser.add_argument("--scaling_params", type=str, default="{}", help="JSON string for scaling method parameters.")
    parser.add_argument("--evaluator_params", type=str, default="{}", help="JSON string for evaluator parameters.")

    # Language penalizer: default is True, use --no-language-penalize to disable
    parser.add_argument('--language_penalize', action=argparse.BooleanOptionalAction, default=True, help="Enable language penalizer by default. Use --no-language-penalize to disable.")
    parser.add_argument("--target_lang", type=str, default="ko", help="Target language code for the penalizer.")

    # Few-shot parameters
    parser.add_argument("--num_few_shot", type=int, default=0, help="Number of few-shot examples to use. Set to 0 to disable.")
    parser.add_argument("--few_shot_split", type=str, default=None, help="Dataset split to draw few-shot examples from (e.g., 'train'). If not set and num_few_shot > 0, examples are taken from the current evaluation split (and excluded from evaluation).")
    parser.add_argument("--few_shot_instruction", type=str, default=DEFAULT_FEW_SHOT_INSTRUCTION, help="Instruction text to prepend to the block of few-shot examples.")
    parser.add_argument("--few_shot_example_template", type=str, default=DEFAULT_FEW_SHOT_EXAMPLE_TEMPLATE, help="Template for formatting each few-shot example. Use {input} and {reference} placeholders.")

    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the global logging level.")

    args = parser.parse_args()

    # Set up global logging level based on CLI argument
    # This will affect all loggers obtained via get_logger if they don't override the level.
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(getattr(logging, args.log_level.upper())) # Update this module's logger level too

    # Parse subset if it's a comma-separated list
    parsed_subset = args.subset
    if args.subset and ',' in args.subset:
        parsed_subset = [s.strip() for s in args.subset.split(',')]
        logger.info(f"Parsed subset list from CLI: {parsed_subset}")

    # Parse JSON string arguments
    try:
        dataset_params = _parse_json_str(args.dataset_params)
        model_params = _parse_json_str(args.model_params)
        judge_params = _parse_json_str(args.judge_params)
        reward_params = _parse_json_str(args.reward_params)
        scaling_params = _parse_json_str(args.scaling_params)
        evaluator_params = _parse_json_str(args.evaluator_params)
    except Exception as e:
        logger.error(f"Error parsing JSON parameters: {e}. Please check your JSON strings.", exc_info=True)
        return # Exit if JSON params are malformed

    # Instantiate Evaluator, passing CLI args as defaults for parameters not directly in run()
    # or parameters that might need to be resolved if run() args are None.
    evaluator_instance = Evaluator(
        default_model_backend=args.model, # More specific than "huggingface" if user provides it
        default_judge_backend=args.judge_model,
        default_reward_backend=args.reward_model,
        default_scaling_method=args.scaling_method,
        default_evaluation_method=args.evaluation_method,
        default_split=args.split,
        default_cot_parser=args.cot_parser, # Default CoT parser path from CLI
        default_num_few_shot=args.num_few_shot,
        default_few_shot_split=args.few_shot_split,
        default_few_shot_instruction=args.few_shot_instruction,
        default_few_shot_example_template=args.few_shot_example_template
    )

    logger.info(f"Starting evaluation for dataset {args.dataset} with model {args.model}")

    # Call run() with parameters directly from CLI args.
    # Evaluator.run will then resolve them against its own defaults if any CLI args are None.
    eval_result = evaluator_instance.run(
        model=args.model, # Pass directly
        judge_model=args.judge_model,
        reward_model=args.reward_model,
        dataset=args.dataset,
        subset=parsed_subset,
        split=args.split,
        scaling_method=args.scaling_method,
        evaluation_method=args.evaluation_method,
        dataset_params=dataset_params,
        model_params=model_params,
        judge_params=judge_params,
        reward_params=reward_params,
        scaling_params=scaling_params,
        evaluator_params=evaluator_params,
        language_penalize=args.language_penalize,
        target_lang=args.target_lang,
        custom_cot_parser=args.cot_parser, # Pass directly
        num_few_shot=args.num_few_shot, # Pass directly
        few_shot_split=args.few_shot_split,
        few_shot_instruction=args.few_shot_instruction,
        few_shot_example_template=args.few_shot_example_template,
    )

    result_dict = eval_result.to_dict()

    # Output results
    output_json_str = json.dumps(result_dict, ensure_ascii=False, indent=2)
    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(output_json_str)
            logger.info(f"Results successfully saved to {args.output_file}")
        except IOError as e:
            logger.error(f"Failed to write results to '{args.output_file}': {e}. Outputting to stdout instead.", exc_info=True)
            print(output_json_str)
    else:
        print(output_json_str)

if __name__ == "__main__":
    main()
