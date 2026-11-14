import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="def add(a, b): return a + b",
)

vector = response.embeddings[0]

print("Embedding dimensions:", len(vector))
print("First five values:", vector[:5])