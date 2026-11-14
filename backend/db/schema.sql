CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repository_chunks (
    id BIGSERIAL PRIMARY KEY,
    repository_url TEXT NOT NULL,
    owner TEXT,
    repository_name TEXT,
    default_branch TEXT,
    commit_sha TEXT,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT repository_chunks_source_unique
        UNIQUE (repository_url, commit_sha, file_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS repository_chunks_embedding_hnsw
    ON repository_chunks
    USING hnsw (embedding vector_cosine_ops);