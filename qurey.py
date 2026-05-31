import chromadb, anthropic, re, hashlib, numpy as np

class Embedder:   # same as ingest.py
    def embed(self, texts, dim=256):
        out = []
        for t in texts:
            v = np.zeros(dim)
            for w in re.findall(r'\b\w+\b', t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(),16) % dim] += 1
            n = np.linalg.norm(v)
            out.append((v/n if n else v).tolist())
        return out

db  = chromadb.PersistentClient(path="./chroma_db")
col = db.get_collection("knowledge_base")
emb = Embedder()
ai  = anthropic.Anthropic()   # needs ANTHROPIC_API_KEY env var

def ask(question, topic=None):
    results = col.query(
        query_embeddings=emb.embed([question]),
        n_results=4,
        where={"topic": topic} if topic else None,
        include=["documents", "metadatas", "distances"]
    )
    context = "\n\n".join(results["documents"][0])
    resp = ai.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user",
            "content": f"Answer using ONLY this context:\n\n{context}\n\nQ: {question}"}]
    )
    return resp.content[0].text

# Ask anything
print(ask("How many leave days do employees get?", topic="hr"))
print(ask("What is the enterprise plan SLA?",       topic="product"))