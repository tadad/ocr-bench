"""Tests for ground-truth CER/WER scoring."""

from __future__ import annotations

import pytest
from datasets import Dataset

from ocr_bench.metrics import levenshtein_distance, normalize_metric_text, score_dataset


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        (["one", "two"], ["one", "too"], 1),
    ],
)
def test_levenshtein_distance(reference, prediction, expected):
    assert levenshtein_distance(reference, prediction) == expected


def test_normalized_text_flattens_html_whitespace_and_unicode():
    decomposed = "Cafe\u0301\r\n<table><tr><td>one</td><td>two</td></tr></table>"
    assert normalize_metric_text(decomposed) == "Café one | two"


def test_raw_text_only_canonicalizes_unicode_and_line_endings():
    assert normalize_metric_text("Cafe\u0301\r\n  two", "raw") == "Café\n  two"


def test_unknown_text_mode_fails():
    with pytest.raises(ValueError, match="Unknown metric text mode"):
        normalize_metric_text("text", "wrong")  # type: ignore[arg-type]


def test_raw_mode_skips_whitespace_only_references():
    dataset = Dataset.from_dict({"reference": [" \n\t"], "ocr": ["text"]})

    result = score_dataset(dataset, {"ocr": "model"}, "reference", text_mode="raw")

    assert result.summaries[0].evaluated_samples == 0
    assert result.summaries[0].skipped_samples == 1
    assert result.details == []


def test_score_dataset_uses_corpus_totals_and_skips_empty_references():
    dataset = Dataset.from_dict(
        {
            "id": ["a", "b", "c"],
            "reference": ["cat", "a much longer line", ""],
            "model_a": ["cut", "a much longer line", "ignored"],
        }
    )

    result = score_dataset(dataset, {"model_a": "model-a"}, "reference")
    summary = result.summaries[0]

    assert summary.char_errors == 1
    assert summary.reference_chars == len("cat") + len("a much longer line")
    assert summary.cer == pytest.approx(1 / summary.reference_chars)
    assert summary.word_errors == 1
    assert summary.reference_words == 5
    assert summary.wer == pytest.approx(0.2)
    assert summary.evaluated_samples == 2
    assert summary.skipped_samples == 1
    assert len(result.details) == 2
    assert result.details[0]["id"] == "a"


def test_failure_sentinel_is_penalized_as_empty_prediction():
    dataset = Dataset.from_dict({"reference": ["three words here"], "ocr": ["[OCR ERROR]"]})
    summary = score_dataset(dataset, {"ocr": "broken-model"}, "reference").summaries[0]

    assert summary.failed_outputs == 1
    assert summary.cer == 1.0
    assert summary.wer == 1.0


def test_scores_multiple_models_and_sorts_by_cer():
    dataset = Dataset.from_dict(
        {
            "reference": ["hello world"],
            "bad": ["hello"],
            "good": ["hello world"],
        }
    )
    result = score_dataset(
        dataset,
        {"bad": "bad-model", "good": "good-model"},
        "reference",
    )

    assert [summary.model for summary in result.summaries] == ["good-model", "bad-model"]
    assert result.summaries[0].cer == 0
    assert result.summaries[0].wer == 0


def test_reference_column_is_not_scored_as_a_model():
    dataset = Dataset.from_dict(
        {"text": ["ground truth"], "ocr_model": ["ground trvth"]}
    )

    result = score_dataset(
        dataset,
        {"text": "text", "ocr_model": "actual-model"},
        "text",
    )

    assert [summary.model for summary in result.summaries] == ["actual-model"]


def test_reference_only_column_map_fails_cleanly():
    dataset = Dataset.from_dict({"text": ["ground truth"]})

    with pytest.raises(ValueError, match="No OCR output columns remain"):
        score_dataset(dataset, {"text": "text"}, "text")


def test_max_samples_limits_scoring():
    dataset = Dataset.from_dict({"reference": ["a", "b"], "ocr": ["x", "b"]})
    summary = score_dataset(dataset, {"ocr": "model"}, "reference", max_samples=1).summaries[0]
    assert summary.evaluated_samples == 1
    assert summary.cer == 1


def test_sampling_matches_judge_indices_and_keeps_source_index():
    dataset = Dataset.from_dict(
        {
            "id": ["zero", "one", "two", "three"],
            "reference": ["a", "b", "c", "d"],
            "ocr": ["a", "b", "c", "d"],
        }
    )
    result = score_dataset(
        dataset, {"ocr": "model"}, "reference", max_samples=2, seed=7
    )

    assert [row["sample_idx"] for row in result.details] == [2, 0]
    assert [row["id"] for row in result.details] == ["two", "zero"]


def test_missing_reference_column_fails_cleanly():
    dataset = Dataset.from_dict({"ocr": ["text"]})
    with pytest.raises(ValueError, match="Reference column 'reference' not found"):
        score_dataset(dataset, {"ocr": "model"}, "reference")
