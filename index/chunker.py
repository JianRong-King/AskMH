import hashlib
import re
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ARTICLES_DIR, CHUNK_MAX_CHARS, CHUNK_OVERLAP, CHUNK_SIZE, ROOT

TABLE_LINE = re.compile(r"^\s*\|")


def load_article(path: Path) -> tuple[dict, str]:
    _, front, body = path.read_text(encoding="utf-8").split("---", 2)
    meta = yaml.safe_load(front)
    body = re.sub(r"^\s*# .*\n", "", body.strip() + "\n", count=1).strip()
    return meta, body


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
                },
            )
        )
    return docs


def load_all_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(ARTICLES_DIR.rglob("*.md")):
        docs.extend(article_to_documents(path))
    return docs
