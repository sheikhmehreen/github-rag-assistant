from pathlib import Path

import pytest

from backend.services import ingestion


def test_validate_github_url_normalizes_git_suffix():
    assert ingestion.validate_github_url("https://github.com/SomeOwner/SomeRepo.git") == \
        "https://github.com/SomeOwner/SomeRepo.git"


@pytest.mark.parametrize("url", [
    "http://github.com/a/b", "https://evil.example/a/b", "https://github.com/a/b/tree/main",
    "https://user@github.com/a/b", "https://github.com:443/a/b", "https://github.com/a/b?x=1",
])
def test_validate_github_url_rejects_unsafe_or_non_repo_urls(url):
    with pytest.raises(ValueError):
        ingestion.validate_github_url(url)


def test_filters_preserve_github_and_dot_configuration(tmp_path):
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".venv\n", encoding="utf-8")
    (tmp_path / ".env.production").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("private key", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\x00binary")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "generated.py").write_text("nope", encoding="utf-8")

    docs = ingestion.collect_repository_files(tmp_path)
    paths = {doc["metadata"]["file_path"] for doc in docs}
    assert paths == {".github/workflows/ci.yml", ".gitignore"}


def test_size_and_count_limits(tmp_path, monkeypatch):
    (tmp_path / "one.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("print(2)\n", encoding="utf-8")
    monkeypatch.setattr(ingestion, "MAX_FILES", 1)
    assert len(ingestion.collect_repository_files(tmp_path)) == 1
    monkeypatch.setattr(ingestion, "MAX_FILES", 10)
    monkeypatch.setattr(ingestion, "MAX_FILE_SIZE", 3)
    assert ingestion.collect_repository_files(tmp_path) == []


def test_ingestion_attaches_repository_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "collect_repository_files", lambda _: [{
        "content": "hello", "metadata": {"file_path": "README.md"}
    }])

    def fake_run(args, **kwargs):
        class Result:
            stdout = "main\n" if "symbolic-ref" in args else "abc123\n"
        return Result()

    monkeypatch.setattr(ingestion, "_run_git", fake_run)
    docs = ingestion.ingest_repository("https://github.com/acme/widget")
    metadata = docs[0]["metadata"]["repository"]
    assert metadata == {
        "url": "https://github.com/acme/widget", "owner": "acme", "name": "widget",
        "default_branch": "main", "commit_sha": "abc123",
    }
