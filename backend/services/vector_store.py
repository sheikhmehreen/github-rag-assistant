"""Save embedded repository chunks in PostgreSQL with pgvector."""

import os

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

EMBEDDING_DIMENSIONS = 768


def _prepare_rows(embedded_chunks):
    """Validate chunks and prepare their values for database insertion."""
    rows = []

    for chunk in embedded_chunks:
        metadata = chunk.get("metadata", {})
        repository = metadata.get("repository", {})
        vector = chunk.get("embedding")

        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Each embedding must be a list of {EMBEDDING_DIMENSIONS} numbers"
            )

        repository_url = repository.get("url")
        file_path = metadata.get("file_path")
        chunk_index = metadata.get("chunk_index")
        content = chunk.get("content")

        if not repository_url or not file_path or chunk_index is None or not content:
            raise ValueError(
                "Each chunk needs repository URL, file path, chunk index, and content"
            )

        rows.append((
            repository_url,
            repository.get("owner"),
            repository.get("name"),
            repository.get("default_branch"),
            repository.get("commit_sha"),
            file_path,
            chunk_index,
            content,
            Vector(vector),
        ))

    return rows


def store_chunks(embedded_chunks, database_url=None):
    """Insert or update embedded chunks; return the number saved."""
    rows = _prepare_rows(embedded_chunks)

    if not rows:
        return 0

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("Set the DATABASE_URL environment variable")

    query = """
        INSERT INTO repository_chunks (
            repository_url,
            owner,
            repository_name,
            default_branch,
            commit_sha,
            file_path,
            chunk_index,
            content,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (
            repository_url,
            commit_sha,
            file_path,
            chunk_index
        )
        DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            owner = EXCLUDED.owner,
            repository_name = EXCLUDED.repository_name,
            default_branch = EXCLUDED.default_branch
    """

    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)

    return len(rows)