"""Create local Ollama embeddings for text chunks."""

import ollama

EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 32


def embed_chunks(chunks, batch_size=DEFAULT_BATCH_SIZE):
    """Return chunks with an embedding vector attached to each one."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    embedded_chunks = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["content"] for chunk in batch]

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        if len(response.embeddings) != len(batch):
            raise RuntimeError("Ollama returned a different number of embeddings than inputs")

        for chunk, vector in zip(batch, response.embeddings):
            embedded_chunks.append({
                "content": chunk["content"],
                "embedding": vector,
                "metadata": chunk["metadata"].copy(),
            })

    return embedded_chunks