import pytest

# Try to import the required modules, skip tests if dependencies are missing
try:
    import pandas as pd
    from llm_eval.datasets.dataset_loader import GenericFileDataset
    GENERIC_FILE_AVAILABLE = True
except ImportError as e:
    GENERIC_FILE_AVAILABLE = False
    pytest.skip(f"Skipping generic file dataset tests due to missing dependencies: {e}", allow_module_level=True)


def test_generic_file_dataset_load_and_raw(tmp_path):
    data = pd.DataFrame({"question": ["Q1", "Q2"], "answer": ["A1", "A2"], "extra": [1, 2]})
    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    ds = GenericFileDataset(
        dataset_name="generic_file",
        file_path=str(csv_path),
        input_col="question",
        reference_col="answer",
    )

    loaded = ds.load()
    assert len(loaded) == 2
    assert loaded[0]["input"] == "Q1"
    assert loaded[0]["reference"] == "A1"
    assert loaded[0]["metadata"]["extra"] == 1

    raw = ds.get_raw_samples()
    assert isinstance(raw, pd.DataFrame)
    assert list(raw.columns) == ["question", "answer", "extra"]


@pytest.mark.parametrize("column_arg", ["input_col", "reference_col"])
def test_missing_column_fails_before_inference(tmp_path, column_arg):
    path = tmp_path / "items.csv"
    path.write_text("input,reference\nq,0\n")
    dataset = GenericFileDataset("generic_file", str(path), **{column_arg: "wrong_column"})
    with pytest.raises(ValueError, match="wrong_column"):
        dataset.load()


def test_blank_reference_in_csv_has_record_number(tmp_path):
    path = tmp_path / "items.csv"
    path.write_text("input,reference\nq,0\nq2,\n")
    with pytest.raises(ValueError, match="record 2"):
        GenericFileDataset("generic_file", str(path)).load()
