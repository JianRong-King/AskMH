import json
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import CHROMA_DIR, CHUNKS_FILE, COLLECTION, RETRIEVE_K
from index.embeddings import get_embeddings

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def load_chunks() -> list[Document]:
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION,
    )


@lru_cache(maxsize=4)
def get_bm25(k: int = RETRIEVE_K) -> BM25Retriever:
    return BM25Retriever.from_documents(load_chunks(), k=k, preprocess_func=tokenize)


@lru_cache(maxsize=4)
def get_retriever(k: int = RETRIEVE_K, dense_weight: float = 0.5):
    """Hybrid dense+BM25, kept for `index/search.py --mode hybrid` and the layer 1 comparison.

    The agent uses dense retrieval instead (see eval/FINDINGS.md F1). langchain_classic is
    imported here rather than at module scope because importing it drags in bs4, lxml and
    tqdm, which the API container would otherwise have to ship for a path it never takes.
    """
    from langchain_classic.retrievers import EnsembleRetriever

    fetch_k = k * 3
    dense = get_vectorstore().as_retriever(search_kwargs={"k": fetch_k})
    return EnsembleRetriever(
        retrievers=[dense, get_bm25(fetch_k)],
        weights=[dense_weight, 1 - dense_weight],
        id_key="url",
    )


def retrieve(query: str, k: int = RETRIEVE_K) -> list[Document]:
    """Dense-only. It beat the hybrid ensemble on both natural and exact-term queries
    across the 50-question eval set; see eval/FINDINGS.md (F1).

    Over-fetch then collapse by article, so a long article cannot occupy several of the
    k slots handed to the grader.
    """
    seen: set[str] = set()
    out: list[Document] = []
    for doc in get_vectorstore().similarity_search(query, k=k * 3):
        url = doc.metadata["url"]
        if url in seen:
            continue
        seen.add(url)
        out.append(doc)
        if len(out) == k:
            break
    return out
