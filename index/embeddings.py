from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_DEVICE, EMBED_MODEL, EMBED_QUERY_INSTRUCTION


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    import torch

    device = EMBED_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": EMBED_QUERY_INSTRUCTION},
    )
