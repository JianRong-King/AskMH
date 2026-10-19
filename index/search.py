import argparse

from agent.retriever import get_bm25, get_retriever, get_vectorstore, retrieve


def main():
    ap = argparse.ArgumentParser(description="Search the AskMH index")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["agent", "hybrid", "dense", "bm25"], default="agent")
    args = ap.parse_args()

    if args.mode == "dense":
        results = [(d, f"{1 - s:.3f}") for d, s in get_vectorstore().similarity_search_with_score(args.query, k=args.k)]
    elif args.mode == "bm25":
        results = [(d, "") for d in get_bm25(args.k).invoke(args.query)]
    elif args.mode == "hybrid":
        results = [(d, "") for d in get_retriever(args.k).invoke(args.query)[:args.k]]
    else:
        results = [(d, "") for d in retrieve(args.query, k=args.k)]

    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        chunk = f" [chunk {m['chunk_index'] + 1}/{m['n_chunks']}]" if m["n_chunks"] > 1 else ""
        print(f"{i}. {m['title']}{chunk}  {score}\n   {m['topic_path']} | {m['url']}")


if __name__ == "__main__":
    main()
