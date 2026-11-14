"""Find the stored repository chunks that best match a question."""

import os

import ollama
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from backend.services.embedding import EMBEDDING_MODEL
from backend.services.vector_store import EMBEDDING_DIMENSIONS


def embed_question(question):
    """Turn a question into a vector using the same model as the chunks."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=question)
    vector = response.embeddings[0]

    if len(vector) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}"
        )

    return vector


def search_chunks(question, repository_url=None, limit=5, database_url=None):
    """Return the chunks closest in meaning to the question, best first."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("Set the DATABASE_URL environment variable")

    query_vector = Vector(embed_question(question.strip()))

    sql = """
        SELECT file_path, chunk_index, content,
               1 - (embedding <=> %s) AS similarity
        FROM repository_chunks
    """
    params = [query_vector]

    if repository_url:
        sql += " WHERE repository_url = %s"
        params.append(repository_url)

    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params.extend([query_vector, limit])

    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    return [
        {
            "file_path": file_path,
            "chunk_index": chunk_index,
            "content": content,
            "similarity": float(similarity),
        }
        for file_path, chunk_index, content, similarity in rows
    ]
