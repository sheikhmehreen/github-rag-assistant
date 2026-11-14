from types import SimpleNamespace

import pytest

from backend.services import embedding


def test_embed_chunks_attaches_vectors_and_preserves_metadata(monkeypatch):
    chunks = [
        {
            "content": "first chunk",
            "metadata": {"file_path": "README.md", "chunk_index": 0},
        },
        {
            "content": "second chunk",
            "metadata": {"file_path": "app.py", "chunk_index": 1},
        },
    ]

    def fake_embed(model, input):
        assert model == "nomic-embed-text"
        return SimpleNamespace(
            embeddings=[[0.1, 0.2], [0.3, 0.4]]
        )

    monkeypatch.setattr(embedding.ollama, "embed", fake_embed)

    result = embedding.embed_chunks(chunks)

    assert [item["embedding"] for item in result] == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert result[0]["content"] == "first chunk"
    assert result[0]["metadata"] == chunks[0]["metadata"]
    assert result[0]["metadata"] is not chunks[0]["metadata"]


def test_embed_chunks_sends_inputs_in_batches(monkeypatch):
    chunks = [
        {"content": f"chunk {number}", "metadata": {"chunk_index": number}}
        for number in range(3)
    ]
    batches = []

    def fake_embed(model, input):
        batches.append(input)
        return SimpleNamespace(embeddings=[[0.5] for _ in input])

    monkeypatch.setattr(embedding.ollama, "embed", fake_embed)

    result = embedding.embed_chunks(chunks, batch_size=2)

    assert batches == [["chunk 0", "chunk 1"], ["chunk 2"]]
    assert len(result) == 3


@pytest.mark.parametrize("batch_size", [0, -1])
def test_embed_chunks_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        embedding.embed_chunks([], batch_size=batch_size)