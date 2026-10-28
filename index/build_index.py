import json
import shutil
import time
from collections import Counter

import chromadb

from config import CHROMA_DIR, CHUNKS_FILE, COLLECTION, EMBED_MODEL
from index.chunker import embedding_text, load_all_documents
from index.embeddings import get_embeddings


def main():
    t0 = time.time()
    docs = load_all_documents()
    per_article = Counter(d.metadata["url"] for d in docs)
    with_questions = len({d.metadata["url"] for d in docs if d.metadata["questions"]})
    print(f"{len(per_article)} articles -> {len(docs)} chunks "
          f"({sum(n > 1 for n in per_article.values())} articles split, "
          f"{with_questions} with doc2query questions)")

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "page_content": d.page_content, "metadata": d.metadata},
                               ensure_ascii=False) + "\n")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    print(f"embedding with {EMBED_MODEL} ...")
    vectors = get_embeddings().embed_documents([embedding_text(d) for d in docs])

    # Embed the question-augmented text but store the plain chunk, so generated questions never
    # reach the grader or generator. Chroma.from_documents can only embed what it stores.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(
        COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )
    collection.add(
        ids=[d.id for d in docs],
        embeddings=vectors,
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
    )
    print(f"done in {time.time() - t0:.1f}s -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
