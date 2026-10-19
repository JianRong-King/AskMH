import json
import shutil
import time
from collections import Counter

from langchain_chroma import Chroma

from config import CHROMA_DIR, CHUNKS_FILE, COLLECTION, EMBED_MODEL
from index.chunker import load_all_documents
from index.embeddings import get_embeddings


def main():
    t0 = time.time()
    docs = load_all_documents()
    per_article = Counter(d.metadata["url"] for d in docs)
    print(f"{len(per_article)} articles -> {len(docs)} chunks "
          f"({sum(n > 1 for n in per_article.values())} articles split)")

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "page_content": d.page_content, "metadata": d.metadata},
                               ensure_ascii=False) + "\n")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    print(f"embedding with {EMBED_MODEL} ...")
    Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        ids=[d.id for d in docs],
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
        collection_metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )
    print(f"done in {time.time() - t0:.1f}s -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
