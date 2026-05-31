import chromadb, re, hashlib, numpy as np
from pathlib import Path

class Embedder:
    def embed(self, texts, dim=256):
        out = []
        for t in texts:
            v = np.zeros(dim)
            for w in re.findall(r'\b\w+\b', t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(),16) % dim] += 1
            n = np.linalg.norm(v)
            out.append((v/n if n else v).tolist())
        return out

db = chromadb.PersistentClient(path="./chroma_db")
col = db.get_or_create_collection("knowledge_base")
emb = Embedder()

def ingest(path, topic):
    text = Path(path).read_text()
    chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
    ids   = [f"{Path(path).stem}_{i}" for i in range(len(chunks))]
    metas = [{"source": Path(path).name, "topic": topic} for _ in chunks]
    col.add(documents=chunks, ids=ids, metadatas=metas, embeddings=emb.embed(chunks))
    print(f"✓ {path} → {len(chunks)} chunks")

# Auto-detect all .md files in docs/ folder
def auto_ingest():
    topic_map = {
        "hr_": "hr",
        "product_": "product",
        "tt": "general",
        # Add more mappings as needed
    }
    
    docs_path = Path("docs")
    for md_file in sorted(docs_path.glob("*.md")):
        # Determine topic based on filename
        topic = "general"
        for prefix, t in topic_map.items():
            if prefix in md_file.name.lower():
                topic = t
                break
        
        ingest(str(md_file), topic=topic)

# Run auto-ingest
auto_ingest()