import asyncio
from unittest.mock import Mock

import pytest

from llm_eval.evaluation.llm_judge import LLMJudgeEvaluator
from llm_eval.models.litellm_backend import LiteLLMBackend
from llm_eval.models.litellm_judge import LiteLLMJudge


def test_judge_attaches_out_of_order_verdicts_to_their_questions(monkeypatch):
    judge = LiteLLMJudge(provider="openai", model_name="test", batch_size=2)
    fast_finished = asyncio.Event()

    async def completion(payload):
        prompt = payload["messages"][0]["content"]
        if "slow-question" in prompt:
            await asyncio.wait_for(fast_finished.wait(), timeout=5)
            return "[[A]]"
        fast_finished.set()
        return "[[B]]"

    monkeypatch.setattr(judge, "_once", completion)
    samples = [{"input": "slow-question", "reference": "A"}, {"input": "fast-question", "reference": "B"}]
    evaluator = LLMJudgeEvaluator(judge, default_judge_type="response_comparison")
    assert evaluator.evaluate_predictions(samples)["judge_accuracy"] == 1
    assert [row["judge_winner"] for row in samples] == ["A", "B"]


def test_generation_preserves_failed_sample_metadata_and_order(monkeypatch):
    model = LiteLLMBackend(provider="openai", model_name="test", batch_size=2)
    fast_finished = asyncio.Event()

    async def completion(payload, **kwargs):
        if payload["messages"][0]["content"] == "slow":
            await asyncio.wait_for(fast_finished.wait(), timeout=5)
            raise TimeoutError("simulated timeout")
        fast_finished.set()
        return "ok"

    monkeypatch.setattr(model, "_generate_with_retry_async", completion)
    samples = [
        {"id": 1, "input": "slow", "reference": "A", "options": ["A", "B"]},
        {"id": 2, "input": "fast", "reference": "B", "metadata": {"split": "test"}},
    ]
    assert model.generate_batch(samples, show_progress=False) == [
        {**samples[0], "prediction": "", "error": "simulated timeout", "finish_reason": "error"},
        {**samples[1], "prediction": "ok"},
    ]


def test_gold_judge_metrics_include_invalid_verdicts():
    judge = Mock()
    judge.judge_batch.return_value = [
        {"prediction": "[[true]]"}, {"prediction": "[[false]]"}, {"prediction": "invalid"},
        {"prediction": "[[true]] and [[false]]"},
        {"prediction": "Score: 4"},
    ]
    samples = [{"input": "q", "reference": "a"} for _ in range(4)]
    samples.append({"judge_type": "k2_eval", "input": "q"})
    evaluator = LLMJudgeEvaluator(judge, default_judge_type="rubric_response_and_gold")
    metrics = evaluator.evaluate_predictions(samples)
    assert metrics["judge_accuracy"] == pytest.approx(1 / 4)
    assert metrics["judge_parse_failures"] == 2
    assert metrics["average_judge_score"] == 4
    assert [s["judge_correct"] for s in samples[:4]] == [True, False, None, None]
    assert "error" in samples[2]["evaluation"]
