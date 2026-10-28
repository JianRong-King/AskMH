import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ARTICLES_DIR, CHUNK_MAX_CHARS, CHUNK_OVERLAP, CHUNK_SIZE, DOC2QUERY_FILE, ROOT

TABLE_LINE = re.compile(r"^\s*\|")


def load_article(path: Path) -> tuple[dict, str]:
    _, front, body = path.read_text(encoding="utf-8").split("---", 2)
    meta = yaml.safe_load(front)
    body = re.sub(r"^\s*# .*\n", "", body.strip() + "\n", count=1).strip()
    return meta, body


def article_hash(meta: dict, body: str) -> str:
    return hashlib.sha1(f"{meta['title']}\n{body}".encode()).hexdigest()


@lru_cache(maxsize=1)
def load_questions() -> dict:
    """index/doc2query.py output: {url: {"hash": ..., "questions": [...]}}."""
    if not DOC2QUERY_FILE.exists():
        return {}
    return json.loads(DOC2QUERY_FILE.read_text(encoding="utf-8"))


def questions_for(meta: dict, body: str) -> list[str]:
    """Generated questions, or none if the article changed since they were generated."""
    entry = load_questions().get(meta["url"])
    return entry["questions"] if entry and entry["hash"] == article_hash(meta, body) else []


def embedding_text(doc: Document) -> str:
    """What gets embedded: the chunk with its article's generated questions after the header.
    page_content stays the article text alone, which is all the grader and generator see."""
    if not doc.metadata.get("questions"):
        return doc.page_content
    header, _, body = doc.page_content.partition("\n\n")
    asked = "\n".join(f"- {q}" for q in doc.metadata["questions"].split("\n"))
    return f"{header}\nAlso asked:\n{asked}\n\n{body}"


def _blocks(body: str) -> list[tuple[str, bool]]:
    blocks: list[tuple[str, bool]] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        is_table = all(TABLE_LINE.match(line) for line in para.splitlines())
        if is_table and blocks and not blocks[-1][1] and len(blocks[-1][0]) < 300:
            prev, _ = blocks.pop()
            blocks.append((f"{prev}\n\n{para}", True))
        else:
            blocks.append((para, is_table))
    return blocks


def split_body(body: str) -> list[str]:
    if len(body) <= CHUNK_MAX_CHARS:
        return [body]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    pieces: list[str] = []
    for text, is_table in _blocks(body):
        pieces.extend([text] if is_table or len(text) <= CHUNK_SIZE else splitter.split_text(text))
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len("\n\n".join(current + [piece])) > CHUNK_SIZE:
            chunks.append("\n\n".join(current))
            tail = current[-1]
            current = [tail] if len(tail) <= CHUNK_OVERLAP else []
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def article_to_documents(path: Path) -> list[Document]:
    meta, body = load_article(path)
    topic_path = " > ".join(meta.get("topic_path") or [meta.get("topic") or "Uncategorized"])
    header = f"Topic: {topic_path}\nQuestion: {meta['title']}\n\n"
    parts = split_body(body) if body else [""]
    questions = "\n".join(questions_for(meta, body))
    docs = []
    for i, part in enumerate(parts):
        docs.append(
            Document(
                id=hashlib.sha1(f"{meta['url']}#{i}".encode()).hexdigest(),
                page_content=header + part,
                metadata={
                    "url": meta["url"],
                    "title": meta["title"],
                    "topic": meta.get("topic") or "Uncategorized",
                    "topic_path": topic_path,
                    "last_updated": meta.get("last_updated") or "",
                    "chunk_index": i,
                    "n_chunks": len(parts),
                    "source_file": path.relative_to(ROOT).as_posix(),
                    "questions": questions,
                },
            )
        )
    return docs


def load_all_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(ARTICLES_DIR.rglob("*.md")):
        docs.extend(article_to_documents(path))
    return docs
