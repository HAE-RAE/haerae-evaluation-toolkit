import pytest

# Try to import the scaling methods module, skip tests if dependencies are missing
try:
    from llm_eval.scaling_methods import SCALING_REGISTRY, load_scaling_method
    from llm_eval.models.base import BaseModel  # Import BaseModel
    # Get scaling method keys from the registry
    scaling_keys = list(SCALING_REGISTRY.keys())
    SCALING_AVAILABLE = True
except ImportError as e:
    SCALING_AVAILABLE = False
    scaling_keys = []
    pytest.skip(f"Skipping scaling tests due to missing dependencies: {e}", allow_module_level=True)

@pytest.mark.parametrize("scaler_key", scaling_keys)
def test_scaler_registration(scaler_key):
    """
    Test scaling method registration.

    This test checks if the scaling method is correctly registered and can be
    instantiated using load_scaling_method().
    """
    scaler = load_scaling_method(scaler_key)
    assert scaler is not None, f"Scaler not found: {scaler_key}"

@pytest.mark.parametrize("scaler_key", scaling_keys)
def test_scaler_apply(scaler_key):
    """
    Test the apply() method of each scaling method.

    This test verifies that the apply() method returns a list and that each item
    in the list contains the 'prediction' key.
    """
    scaler = load_scaling_method(scaler_key)

    # Mock model for testing (using BaseModel as a base)
    class MockModel(BaseModel):
        def generate_batch(self, inputs, return_logits=False, **kwargs):
            output = {"prediction": "Hello world"}
            if return_logits:
                output["logits"] = {"sum_log_prob": -1.0}
            return [dict(output) for _ in inputs]

    mock_model = MockModel()
    scaler.model = mock_model  # Assign the mock model to the scaler

    test_data = [{"input": "Test 1", "reference": "Ref 1"}]
    try:
        output_data = scaler.apply(test_data)
        assert isinstance(output_data, list), "apply() should return a list."
        assert len(output_data) == len(test_data), "Output should have the same length as input."
        for item in output_data:
            assert "prediction" in item, "Each item should have 'prediction' key."
    except Exception as e:
        pytest.fail(f"Scaler {scaler_key} failed during apply() call: {e}")

@pytest.mark.parametrize("use_reward", [True, False])
def test_best_of_n_uses_configured_scores(use_reward):
    from llm_eval.models.multi import MultiModel

    class Generator(BaseModel):
        def __init__(self):
            self.round = 0

        def generate_batch(self, inputs, return_logits=False, **kwargs):
            assert return_logits == (not use_reward)
            self.round += 1
            results = []
            for item in inputs:
                score = -abs(self.round - item["best_round"])
                result = {**item, "prediction": str(self.round)}
                if return_logits:
                    result["logits"] = {"sum_log_prob": score}
                results.append(result)
            return results

    class Reward:
        def score_batch(self, items):
            return [{**row, "reward": -abs(int(row["prediction"]) - row["best_round"])} for row in items]

    model = MultiModel()
    model.generate_model = Generator()
    if use_reward:
        model.reward_model = Reward()
    scaler = load_scaling_method("best_of_n", model=model, n=3, batch_size=2)
    outputs = scaler.apply([{"input": "x", "best_round": 3}, {"input": "y", "best_round": 2}])
    assert [row["prediction"] for row in outputs] == ["3", "2"]


def test_best_of_n_reports_unavailable_scores():
    class Unscored(BaseModel):
        def generate_batch(self, items, **kwargs):
            return [{**row, "prediction": "a"} for row in items]

    scaler = load_scaling_method("best_of_n", model=Unscored())
    with pytest.raises(ValueError, match="logits.sum_log_prob"):
        scaler.apply([{"input": "q"}])
