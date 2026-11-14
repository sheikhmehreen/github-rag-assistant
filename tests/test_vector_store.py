import pytest

from backend.services.vector_store import _prepare_rows


def make_embedded_chunk():
    return {
        "content": "A test chunk",
        "embedding": [0.0] * 768,
        "metadata": {
            "file_path": "README.md",
            "chunk_index": 0,
            "repository": {
                "url": "https://github.com/acme/demo",
                "owner": "acme",
                "name": "demo",
                "default_branch": "main",
                "commit_sha": "abc123",
            },
        },
    }


def test_prepare_rows_keeps_chunk_and_repository_details():
    rows = _prepare_rows([make_embedded_chunk()])

    assert len(rows) == 1
    assert rows[0][0] == "https://github.com/acme/demo"
    assert rows[0][5] == "README.md"
    assert rows[0][6] == 0
    assert rows[0][7] == "A test chunk"
    assert len(rows[0][8]) == 768


def test_prepare_rows_accepts_empty_input():
    assert _prepare_rows([]) == []


def test_prepare_rows_rejects_wrong_embedding_dimension():
    chunk = make_embedded_chunk()
    chunk["embedding"] = [0.0] * 10

    with pytest.raises(ValueError, match="768"):
        _prepare_rows([chunk])


def test_prepare_rows_rejects_missing_file_path():
    chunk = make_embedded_chunk()
    del chunk["metadata"]["file_path"]

    with pytest.raises(ValueError, match="file path"):
        _prepare_rows([chunk])