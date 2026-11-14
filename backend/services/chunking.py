"""Split repository documents into overlapping chunks."""


def chunk_documents(documents, chunk_size=1000, overlap=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")

    chunks = []

    for document in documents:
        content = document.get("content", "")
        metadata = document.get("metadata", {})

        if not content:
            continue

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                },
            })

            if end == len(content):
                break

            start = end - overlap
            chunk_index += 1

    return chunks