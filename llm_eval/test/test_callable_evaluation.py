import json
from unittest.mock import Mock

from llm_eval import hret
from llm_eval.models import MODEL_REGISTRY
from llm_eval.models.base import BaseModel
from llm_eval.runner import PipelineRunner


def file_config(tmp_path):
    path = tmp_path / "items.csv"
    path.write_text("input,reference\nfirst,first\nsecond,second\n")
    return {
        "dataset_params": {"file_path": str(path), "dataset_name": "generic_file"},
        "language_penalize": False,
    }


def test_public_callable_entry_points(tmp_path, monkeypatch):
    monkeypatch.setattr(hret._global_config, "auto_save_results", False)
    config = file_config(tmp_path)
    identity = Mock(side_effect=lambda prompt: prompt)
    result = hret.quick_eval(identity, dataset="generic_file", **config)
    assert [call.args[0] for call in identity.call_args_list] == ["first", "second"]
    assert result.metrics["accuracy"] == 1

    @hret.evaluate(dataset="generic_file", **config)
    def wrong(prompt):
        return "wrong"

    assert wrong().metrics["accuracy"] == 0
    results = hret.compare_models(
        {"identity": identity, "wrong": lambda text: "wrong"}, dataset="generic_file", **config
    )
    assert {name: r.metrics["accuracy"] for name, r in results.items()} == {"identity": 1, "wrong": 0}


def test_credentials_stay_out_of_logs_and_tracking(tmp_path, monkeypatch, caplog):
    secret = "test-only-provider-credential"
    received = []

    class Backend(BaseModel):
        def __init__(self, **kwargs):
            received.append(kwargs)

    monkeypatch.setitem(MODEL_REGISTRY, "test_backend", Backend)
    params = {"generate_model": {"name": "test_backend", "params": {"extra_headers": {"Authorization": secret}}}}
    config = file_config(tmp_path)
    PipelineRunner("generic_file", model_backend_name="multi", model_backend_params=params, **config)
    assert received == [params["generate_model"]["params"]]
    assert secret not in caplog.text

    monkeypatch.setattr(hret._global_config, "output_dir", str(tmp_path))
    tracker = Mock()
    with hret.evaluation_context("generic_file", "multi", model_backend_params=params, scaling_params={"n": 3, "custom": secret}, **config) as ctx:
        ctx.add_mlops_integration(tracker)
        ctx.save_results("result.json")
    assert secret not in (tmp_path / "result.json").read_text()
    tracked = tracker.call_args.args[0]["metadata"]["config"]
    assert tracked["dataset_name"] == "generic_file"
    assert tracked["scaling_params"] == {"n": 3}
    assert secret not in json.dumps(tracked)


def test_invalid_cli_json_does_not_log_raw_credentials(caplog):
    from llm_eval.evaluator import _parse_json_str

    assert _parse_json_str('{"api_key": "dummy-cli-secret",}') == {}
    assert "dummy-cli-secret" not in caplog.text
