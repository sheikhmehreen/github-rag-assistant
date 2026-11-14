"""Try retrieval by hand: python tests/check_retrieval.py "your question" """
import sys

from backend.services.retrieval import search_chunks

question = " ".join(sys.argv[1:]) or "How are repositories cloned safely?"
results = search_chunks(question, limit=5)

print(f"Question: {question}\n")
for rank, item in enumerate(results, start=1):
    preview = item["content"][:150].replace("\n", " ")
    print(f"{rank}. {item['file_path']} (chunk {item['chunk_index']}) "
          f"similarity={item['similarity']:.3f}")
    print(f"   {preview}\n")
