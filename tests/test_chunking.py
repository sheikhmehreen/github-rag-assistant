import pytest

from backend.services.chunking import chunk_documents


def test_short_document_produces_one_unchanged_chunk():
    documents = [{"content": "Hello, world!", "metadata": {"file_path": "README.md"}}]

    chunks = chunk_documents(documents, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0]["content"] == "Hello, world!"


def test_long_document_chunks_have_expected_overlap():
    documents = [{"content": "abcdefghij", "metadata": {}}]

    chunks = chunk_documents(documents, chunk_size=6, overlap=2)

    assert [chunk["content"] for chunk in chunks] == ["abcdef", "efghij"]
    assert chunks[0]["content"][-2:] == chunks[1]["content"][:2]


def test_chunks_keep_metadata_and_have_increasing_indices():
    documents = [{
        "content": "abcdefghij",
        "metadata": {
            "file_path": "src/app.py",
            "repository": {"owner": "acme", "name": "demo"},
        },
    }]

    chunks = chunk_documents(documents, chunk_size=6, overlap=2)

    assert [chunk["metadata"]["chunk_index"] for chunk in chunks] == [0, 1]
    for chunk in chunks:
        assert chunk["metadata"]["file_path"] == "src/app.py"
        assert chunk["metadata"]["repository"] == {"owner": "acme", "name": "demo"}


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),    # chunk size must be positive
        (-1, 0),   # chunk size cannot be negative
        (10, -1),  # overlap cannot be negative
        (10, 10),  # overlap must be smaller than chunk size
        (10, 11),
    ],
)
def test_invalid_chunk_settings_raise_value_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=chunk_size, overlap=overlap)