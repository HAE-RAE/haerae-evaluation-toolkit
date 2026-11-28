from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionTokenLogprob
from openai.types.chat.chat_completion import ChoiceLogprobs

from llm_eval.models.openai_backend import OpenAIModel
from llm_eval.scaling_methods.best_of_n import BestOfN


@pytest.mark.parametrize("chat", [True, False])
def test_best_of_n_requests_and_uses_openai_scores(chat, monkeypatch):
    model = OpenAIModel(
        model_name="test", api_base="https://example.invalid/v1",
        api_key="test", use_chat_api=chat, max_retries=0,
    )
    responses = []
    for text, score in [("first", -2), ("second", -1)]:
        if chat:
            token = ChatCompletionTokenLogprob(token=text, logprob=score, top_logprobs=[])
            choice = SimpleNamespace(message=SimpleNamespace(content=text), logprobs=ChoiceLogprobs(content=[token]))
        else:
            choice = SimpleNamespace(text=text, logprobs=SimpleNamespace(token_logprobs=[score], tokens=[text]))
        choice.finish_reason = "stop"
        responses.append(SimpleNamespace(choices=[choice]))
    create = AsyncMock(side_effect=responses)
    endpoint = model.client.chat.completions if chat else model.client.completions
    monkeypatch.setattr(endpoint, "create", create)
    rows = BestOfN(model, n=2).apply([{"input": "question"}])
    assert rows[0]["prediction"] == "second"
    for call in create.await_args_list:
        if chat:
            assert call.kwargs["logprobs"] is True
        else:
            assert type(call.kwargs["logprobs"]) is int
            assert call.kwargs["logprobs"] == 1
