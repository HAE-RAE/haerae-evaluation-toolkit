import pytest

try:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    from llm_eval.models.huggingface_backend import HuggingFaceModel
except ImportError as e:
    pytest.skip(
        f"Skipping HuggingFace backend tests due to missing dependencies: {e}",
        allow_module_level=True,
    )


WORDS = ["<pad>", "<eos>", "a", "b", "c", "d", "e", "x", "y", "z"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A tiny random Llama model and word-level tokenizer, saved locally."""
    path = tmp_path_factory.mktemp("tiny-llama")
    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        model_max_length=64,
    ).save_pretrained(path)
    config = LlamaConfig(
        vocab_size=len(WORDS),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def test_batched_generation_excludes_prompt_tokens(tiny_model_dir,
                                                   monkeypatch):
    """With batch_size > 1 and left padding, predictions hold only new tokens.

    Shorter prompts in a batch are left-padded, so the generated tokens of
    every row start at the padded prompt width.
    """
    model = HuggingFaceModel(
        model_name_or_path=tiny_model_dir, device="cpu", do_sample=False
    )
    new_ids = torch.tensor([7, 8, 9])  # "x y z"

    def fake_generate(input_ids=None, **kwargs):
        appended = new_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
        return torch.cat([input_ids, appended], dim=1)

    monkeypatch.setattr(model.model, "generate", fake_generate)

    inputs = [{"input": "a"}, {"input": "a b c d e"}, {"input": "b c"}]
    results = model.generate_batch(inputs, batch_size=3, show_progress=False)

    assert [r["prediction"] for r in results] == ["x y z"] * 3


def test_mixed_batch_preserves_sample_order(tiny_model_dir, monkeypatch):
    model = HuggingFaceModel(model_name_or_path=tiny_model_dir, device="cpu")
    monkeypatch.setattr(model, "_score_option", lambda prompt, option: -1.0)
    monkeypatch.setattr(model, "_generate_normal", lambda items, *args, **kwargs: [
        {**item, "prediction": "generated"} for item in items
    ])
    rows = [{"input": "a"}, {"input": "b", "options": ["x"]}, {"input": "c"}]
    results = model.generate_batch(rows, batch_size=3, return_logits=True, show_progress=False)
    assert results[1]["logits"]["sum_log_prob"] == -1.0
    assert [(row["input"], row["prediction"]) for row in results] == [
        ("a", "generated"), ("b", "x"), ("c", "generated")
    ]
