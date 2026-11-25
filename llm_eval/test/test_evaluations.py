import json

import pytest

# Try to import the evaluation module, skip tests if dependencies are missing
try:
    from llm_eval.evaluation import EVALUATION_REGISTRY, get_evaluator
    from llm_eval.models.base import BaseModel  # Import BaseModel
    # Get evaluator keys from the registry
    eval_keys = list(EVALUATION_REGISTRY.keys())
    EVALUATION_AVAILABLE = True
except ImportError as e:
    EVALUATION_AVAILABLE = False
    eval_keys = []
    pytest.skip(f"Skipping evaluation tests due to missing dependencies: {e}", allow_module_level=True)

@pytest.mark.parametrize("eval_key", eval_keys)
def test_evaluator_registration(eval_key):
    """
    Test evaluator registration and instance creation.

    This test checks if the evaluator is correctly registered and can be
    instantiated using get_evaluator().
    """
    if eval_key == "llm_judge":
        pytest.skip("Skipping test.")
    evaluator = get_evaluator(eval_key)
    assert evaluator is not None, f"Evaluator not found: {eval_key}"

@pytest.mark.parametrize("eval_key", eval_keys)
def test_evaluator_evaluate(eval_key):
    if eval_key == "llm_judge":
        pytest.skip("Skipping test.")
    """
    Test the evaluate() method of each evaluator.

    This test verifies that the evaluate() method returns a dictionary
    containing 'metrics' and 'samples' keys.
    """
    evaluator = get_evaluator(eval_key)

    # Mock model for testing (using BaseModel as a base)
    class MockModel(BaseModel):
        def generate_batch(self, inputs, return_logits=False, **kwargs):
            return [{"prediction": "test prediction"} for _ in inputs]

    mock_model = MockModel()

    # Test data
    test_data = [
        {"input": "test input", "reference": "test reference", "prediction": "test prediction"}
    ]
    try:
        results = evaluator.evaluate(test_data, model=mock_model)
        assert isinstance(results, dict), "evaluate() should return a dictionary."
        assert "metrics" in results, "Results should contain 'metrics' key."
        assert "samples" in results, "Results should contain 'samples' key."
    except Exception as e:
        pytest.fail(f"Evaluator {eval_key} failed during evaluate() call: {e}")

def test_math_match_plain_reference_and_cot_prediction():
    """math_match credits a correct \\boxed{} answer against a plain reference.

    HRM8K references are bare numbers such as "18.0", and its prompt asks for
    the final answer in \\boxed{} after the reasoning.
    """
    evaluator = get_evaluator("math_match")
    samples = [
        {"prediction": "풀이: 9 * 2 = 18\n\\boxed{18}", "reference": "18.0"},
        {"prediction": "풀이: 9 * 2 = 17\n\\boxed{17}", "reference": "18.0"},
        {"prediction": "\\boxed{3}", "reference": "3"},
    ]
    metrics = evaluator.evaluate_predictions(samples)
    assert [s["evaluation"]["is_correct"] for s in samples] == [True, False, True]
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["parse_failure_rate"] == 0.0
    json.dumps(samples)  # per-sample records must stay JSON serializable


def test_math_match_counts_unparseable_prediction_as_parse_failure():
    evaluator = get_evaluator("math_match")
    samples = [{"prediction": "모르겠습니다", "reference": "18.0"}]
    metrics = evaluator.evaluate_predictions(samples)
    assert metrics["accuracy"] == 0.0
    assert metrics["parse_failure_rate"] == 1.0
