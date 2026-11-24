"""Regression tests for chunking long transcript spans before embedding."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from app.pipeline import embeddings as E

LONG_SPAN = " ".join(
    f"Sentence {i} explains how a router forwards packets between local and wide area networks."
    for i in range(120)
)


class RecordingEncoder:
    """Fake encoder that records exact submitted text and returns stable vectors."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray(
            [[1.0, float(len(text) % 7 + 1), 0.5, 0.25][: self.dim] for text in texts],
            dtype=np.float32,
        )


def test_short_spans_are_not_split():
    assert E.split_span("A short transcript sentence.") == ["A short transcript sentence."]
    assert E.split_span("   ") == []


def test_long_span_is_split_at_sentence_boundaries():
    assert len(LONG_SPAN) > 6000
    chunks = E.split_span(LONG_SPAN)
    assert len(chunks) > 1
    assert all(len(chunk) <= E.MAX_SPAN_CHARS for chunk in chunks)
    # No chunk starts in the middle of a sentence.
    assert all(chunk.startswith("Sentence ") for chunk in chunks)
    assert "".join(chunk.replace(" ", "") for chunk in chunks) == LONG_SPAN.replace(" ", "")


def test_unusually_long_sentence_is_split_safely():
    single = "word " * 800 + "tail"
    chunks = E.split_span(single)
    assert len(chunks) > 1
    assert all(len(chunk) <= E.MAX_SPAN_CHARS for chunk in chunks)
    assert "".join(chunk.replace(" ", "") for chunk in chunks) == single.replace(" ", "")

    one_word = "x" * 2500
    assert all(len(chunk) <= E.MAX_SPAN_CHARS for chunk in E.split_span(one_word))


def test_eight_long_spans_never_produce_a_6000_character_request():
    encoder = RecordingEncoder()
    spans = [LONG_SPAN] * 8
    vectors = E.encode_long_texts(encoder, spans)

    assert vectors.shape == (8, 4)
    assert len(encoder.calls) == 1, "all chunks go through one batched encode call"
    assert len(set(encoder.calls[0])) == len(E.split_span(LONG_SPAN)), "duplicate chunks are deduplicated"
    assert max(len(text) for text in encoder.calls[0]) <= E.MAX_SPAN_CHARS
    assert all(len(text) <= E.MAX_SPAN_CHARS for text in encoder.calls[0])
    assert np.all(np.isfinite(vectors))
    assert all(math.isclose(float(np.linalg.norm(row)), 1.0, abs_tol=1e-5) for row in vectors)


def test_pooled_vectors_keep_dimensions_and_length_weighting():
    chunks = E.split_span(LONG_SPAN)
    assert len(chunks) > 1

    class DistinctChunkEncoder:
        """Deterministic: each chunk gets a different vector, so weighting matters."""

        def __init__(self):
            self.calls: list[list[str]] = []
            self.by_text = {
                chunk: np.array([float(i + 1), float((i % 3) + 1), 1.0, 0.5], dtype=np.float32)
                for i, chunk in enumerate(chunks)
            }

        def encode(self, texts: list[str]) -> np.ndarray:
            self.calls.append(list(texts))
            return np.stack([self.by_text[text] for text in texts])

    encoder = DistinctChunkEncoder()
    vectors = E.encode_long_texts(encoder, [LONG_SPAN])
    assert vectors.shape == (1, 4)
    assert np.all(np.isfinite(vectors))
    assert math.isclose(float(np.linalg.norm(vectors[0])), 1.0, abs_tol=1e-5)

    # Expected: length-weighted mean, then L2-normalized.
    raw = np.stack([encoder.by_text[c] for c in chunks])
    weights = np.asarray([len(c) for c in chunks], dtype=np.float32)
    weighted_mean = (raw * weights[:, None]).sum(axis=0) / weights.sum()
    expected = weighted_mean / np.linalg.norm(weighted_mean)
    np.testing.assert_allclose(vectors[0], expected, atol=1e-6)

    # An equal-weight implementation must fail this test.
    equal_mean = raw.mean(axis=0)
    equal_expected = equal_mean / np.linalg.norm(equal_mean)
    assert not np.allclose(vectors[0], equal_expected, atol=1e-6)

    # A duplicate span must pool identically, proving chunk order is stable.
    again = E.encode_long_texts(DistinctChunkEncoder(), [LONG_SPAN])
    np.testing.assert_allclose(vectors, again)


def test_faithfulness_scores_stay_in_range_with_long_spans():
    from app.pipeline.builder import score_faithfulness

    encoder = E.Embedder(["router packet forwarding"])
    nodes = [
        {"id": "a", "type": "concept", "text": "Router forwarding"},
        {"id": "b", "type": "detail", "text": "Packets cross networks"},
    ]
    score_faithfulness(nodes, encoder, {"a": LONG_SPAN, "b": LONG_SPAN[:2000]})
    for node in nodes:
        assert 0.0 <= node["faith"] <= 1.0


def test_chunk_cache_is_reused_across_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "_CACHE_DIR", tmp_path / "chunk-cache")
    monkeypatch.setattr(E.settings, "embedding_local_model", "test-local-model")

    backend_calls: list[list[str]] = []

    def backend(texts: list[str]) -> np.ndarray:
        backend_calls.append(list(texts))
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    class CachedEncoder:
        def encode(self, texts: list[str]) -> np.ndarray:
            return E.cached_encode(texts, backend)

    encoder = CachedEncoder()
    first = E.encode_long_texts(encoder, [LONG_SPAN])
    calls_after_first = len(backend_calls)
    assert calls_after_first >= 1
    assert len(E.split_span(LONG_SPAN)) > 1, "a long span must produce several chunks"
    second = E.encode_long_texts(encoder, [LONG_SPAN])
    assert len(backend_calls) == calls_after_first, "repeated run must hit the durable cache"
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("language,title", [
    ("en", "Networking in English"),
    ("hi", "नेटवर्किंग"),
    ("hi", "Networking Hinglish"),
])
def test_multilingual_skeletons_still_ground_long_spans(language, title):
    from test_pipeline import synthetic_transcript
    from app.pipeline.builder import build_skeleton

    transcript = synthetic_transcript()
    transcript[0]["text"] = transcript[0]["text"] + " " + LONG_SPAN[:3000]
    mindmap = build_skeleton({
        "videoId": "LONGSPAN01", "title": title, "transcript": transcript,
        "mode": "deep", "teachingLanguage": language, "useLLM": False, "allowWhisper": False,
    })
    concepts = [c for s in mindmap["root"]["children"] for c in s.get("children", [])
                if c.get("type") == "concept"]
    assert concepts
    assert all(0.0 <= c.get("faith", 0.0) <= 1.0 for c in concepts)


# ---------------------------------------------------------------------------
# Acceptance verification: distinct spans, honest dedup, faithfulness parity,
# Devanagari punctuation and invalid-size handling
# ---------------------------------------------------------------------------
TOPICS = ("router", "switch", "fibre", "wireless", "subnet", "firewall", "dns", "tcpip")


def distinct_spans() -> list[str]:
    """Eight genuinely different 6,000-character transcript spans.

    Every sentence is unique, so deduplication cannot artificially shrink the
    workload: each span contributes its own chunks.
    """
    spans = []
    for index, topic in enumerate(TOPICS):
        sentences = [
            f"Lesson {index} point {number} explains how the {topic} device handles traffic differently."
            for number in range(90)
        ]
        spans.append(" ".join(sentences)[:6000])
    return spans


def test_eight_distinct_6000_character_spans_benchmark():
    spans = distinct_spans()
    assert len(spans) == 8
    assert len(set(spans)) == 8, "benchmark must not reuse the same span"
    assert {len(span) for span in spans} == {6000}

    encoder = RecordingEncoder()
    vectors = E.encode_long_texts(encoder, spans)
    submitted = encoder.calls[0]

    assert vectors.shape == (8, 4)
    assert all(len(text) <= E.MAX_SPAN_CHARS for text in submitted)
    assert len(set(submitted)) == len(submitted), "distinct spans must not be deduplicated away"
    assert len(submitted) >= 8 * 6, "each distinct span must contribute several chunks"
    assert np.all(np.isfinite(vectors))
    assert all(math.isclose(float(np.linalg.norm(row)), 1.0, abs_tol=1e-5) for row in vectors)


def test_dedup_removes_only_genuine_duplicates():
    shared = "Routers forward packets between networks. " * 30
    fibre = "Fibre cables carry data beneath oceans. " * 30
    switching = "Switches connect devices inside a LAN. " * 30
    spans = [shared + fibre, shared + switching]
    encoder = RecordingEncoder()
    E.encode_long_texts(encoder, spans)

    expected = set(E.split_span(spans[0]) + E.split_span(spans[1]))
    assert set(encoder.calls[0]) == expected
    assert len(encoder.calls[0]) == len(expected), "only true duplicates may be removed"


class BagOfWordsEncoder:
    """Deterministic lexical encoder so relevance ranking is well-defined."""

    VOCAB = ("router", "forward", "packet", "network", "tcp", "ip", "weather", "sunny")

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            counts = np.asarray([lower.count(word) for word in self.VOCAB], dtype=np.float32)
            norm = float(np.linalg.norm(counts)) or 1.0
            vectors.append(counts / norm)
        return np.stack(vectors)


def test_full_span_and_pooled_faithfulness_agree_on_short_inputs():
    span = "A router forwards packets between a local network and a wide area network using TCP/IP."
    relevant = "Router forwarding packets TCP/IP"
    unrelated = "The weather is warm and sunny today"

    encoder = BagOfWordsEncoder()
    full = encoder.encode([span])
    pooled = E.encode_long_texts(encoder, [span])
    assert E.split_span(span) == [span], "representative input must be a single chunk"
    np.testing.assert_allclose(full, pooled, atol=1e-5)

    relevant_vector = encoder.encode([relevant])
    unrelated_vector = encoder.encode([unrelated])
    full_relevant = float(full[0] @ relevant_vector[0])
    full_unrelated = float(full[0] @ unrelated_vector[0])
    pooled_relevant = float(pooled[0] @ relevant_vector[0])
    pooled_unrelated = float(pooled[0] @ unrelated_vector[0])
    assert full_relevant > full_unrelated
    assert pooled_relevant > pooled_unrelated
    assert math.isclose(full_relevant, pooled_relevant, abs_tol=1e-5)


HINDI_SPAN = "राउटर पैकेट को एक नेटवर्क से दूसरे नेटवर्क तक ले जाता है। " * 45


def test_devanagari_sentence_punctuation_is_respected():
    chunks = E.split_span(HINDI_SPAN)
    assert len(chunks) > 1
    assert all("।" in chunk for chunk in chunks), "splits must follow Devanagari danda boundaries"
    assert all(len(chunk) <= E.MAX_SPAN_CHARS for chunk in chunks)
    vectors = E.encode_long_texts(RecordingEncoder(), [HINDI_SPAN])
    assert vectors.shape == (1, 4)
    assert np.all(np.isfinite(vectors))
    assert math.isclose(float(np.linalg.norm(vectors[0])), 1.0, abs_tol=1e-5)


def test_hindi_transcript_with_danda_builds_grounded_map():
    from app.pipeline.builder import build_skeleton

    transcript = [
        {"start": 0.0, "end": 30.0, "text": HINDI_SPAN[:600]},
        {"start": 30.0, "end": 60.0, "text": "लैन एक इमारत के अंदर devices को जोड़ती है। WAN कई शहरों को जोड़ता है।"},
        {"start": 60.0, "end": 90.0, "text": "स्विच कंप्यूटरों को आपस में जोड़ता है। फायरवॉल ट्रैफ़िक को नियंत्रित करता है।"},
    ]
    mindmap = build_skeleton({
        "videoId": "HINDIDANDA1", "title": "नेटवर्किंग", "transcript": transcript,
        "mode": "deep", "teachingLanguage": "hi", "useLLM": False, "allowWhisper": False,
    })
    assert mindmap["root"]["children"]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "900", None])
def test_non_positive_or_invalid_chunk_size_is_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        E.split_span("some transcript text", bad)
    with pytest.raises(ValueError, match="positive integer"):
        E.encode_long_texts(RecordingEncoder(), ["some transcript text"], bad)
