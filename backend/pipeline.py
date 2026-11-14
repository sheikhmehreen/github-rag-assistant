"""Run the full ingestion pipeline: GitHub URL -> vector database."""
from dotenv import load_dotenv
load_dotenv()
import sys

from backend.services.chunking import chunk_documents
from backend.services.embedding import embed_chunks
from backend.services.ingestion import ingest_repository
from backend.services.vector_store import delete_old_commits, store_chunks


def run_pipeline(repo_url):
    documents = ingest_repository(repo_url)
    print(f"Read {len(documents)} files")
    chunks = chunk_documents(documents)
    print(f"Made {len(chunks)} chunks")
    embedded = embed_chunks(chunks)
    saved = store_chunks(embedded)
    print(f"Stored {saved} chunks")
    if embedded:
        repo = embedded[0]["metadata"]["repository"]
        removed = delete_old_commits(repo["url"], repo["commit_sha"])
        print(f"Removed {removed} chunks from older commits")
    return saved


if __name__ == "__main__":
    run_pipeline(sys.argv[1])