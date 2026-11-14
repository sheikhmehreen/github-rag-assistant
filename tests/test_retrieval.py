from types import SimpleNamespace
from unittest import mock

import pytest

from backend.services import retrieval


def fake_connection(rows, captured):
    cursor = mock.MagicMock()
    cursor.fetchall.return_value = rows
    cursor.execute.side_effect = lambda sql, params: captured.update(sql=sql, params=params)
    connection = mock.MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection


def test_search_returns_ranked_chunks_and_filters_repository(monkeypatch):
    captured = {}
    rows = [("a.py", 0, "code A", 0.91), ("b.py", 2, "code B", 0.80)]
    monkeypatch.setattr(retrieval.ollama, "embed",
                        lambda **kw: SimpleNamespace(embeddings=[[0.1] * 768]))
    monkeypatch.setattr(retrieval.psycopg, "connect",
                        lambda url: fake_connection(rows, captured))
    monkeypatch.setattr(retrieval, "register_vector", lambda conn: None)

    results = retrieval.search_chunks(
        "how does it work?", repository_url="https://github.com/acme/demo",
        limit=2, database_url="postgresql://x",
    )

    assert [r["file_path"] for r in results] == ["a.py", "b.py"]
    assert results[0]["similarity"] == 0.91
    assert "WHERE repository_url = %s" in captured["sql"]
    assert captured["params"][1] == "https://github.com/acme/demo"
    assert captured["params"][-1] == 2


def test_search_rejects_blank_question_and_bad_limit():
    with pytest.raises(ValueError):
        retrieval.search_chunks("   ", database_url="postgresql://x")
    with pytest.raises(ValueError):
        retrieval.search_chunks("hi", limit=0, database_url="postgresql://x")


def test_embed_question_rejects_wrong_dimensions(monkeypatch):
    monkeypatch.setattr(retrieval.ollama, "embed",
                        lambda **kw: SimpleNamespace(embeddings=[[0.1] * 10]))
    with pytest.raises(RuntimeError):
        retrieval.embed_question("hi")
