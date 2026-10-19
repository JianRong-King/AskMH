import argparse
import asyncio
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from playwright.async_api import async_playwright
from tqdm import tqdm

from ingest.sitemap import BASE, fetch_article_urls, slug_from_url
from ingest.to_markdown import html_to_markdown

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "articles"
MANIFEST = ROOT / "data" / "manifest.json"
FAILED = ROOT / "data" / "failed.json"

DATE_RE = re.compile(r"\b(\d{1,2} [A-Z][a-z]{2} \d{4})(?!\d)")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "uncategorized"


class LoginRequired(Exception):
    pass


def _fmt_date(iso: str) -> str | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%d %b %Y").lstrip("0")
    except ValueError:
        return None


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


async def extract(page, url: str) -> dict:
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_function(
        "() => location.pathname.startsWith('/faq/login') || document.querySelector('.article-head')",
        timeout=20000,
    )
    if "/faq/login" in page.url:
        raise LoginRequired("article requires login (not public)")
    await page.wait_for_timeout(1500)
    title = (await page.inner_text(".article-head")).strip()
    bodies = await page.eval_on_selector_all(
        ".selfServiceArticleLayout .slds-rich-text-editor__output", "els => els.map(e => e.outerHTML)"
    )
    topics = await page.eval_on_selector_all(
        "nav[aria-label='Breadcrumbs'] a",
        "els => els.map(e => ({name: e.getAttribute('title') || e.innerText.trim(), href: e.getAttribute('href')}))",
    )
    layout_text = await page.inner_text(".selfServiceArticleLayout")
    date = DATE_RE.search(layout_text)
    return {
        "title": title,
        "body_html": "\n".join(bodies),
        "topics": [t for t in topics if t["name"]],
        "last_updated": date.group(1) if date else None,
    }


def write_markdown(item: dict, data: dict) -> Path:
    topics = data["topics"]
    topic = topics[-1]["name"] if topics else "Uncategorized"
    body = html_to_markdown(data["body_html"]) if data["body_html"] else ""
    front = {
        "title": data["title"],
        "url": item["url"],
        "topic": topic,
        "topic_path": [t["name"] for t in topics],
        "topic_url": BASE + topics[-1]["href"] if topics and topics[-1]["href"] else None,
        "last_updated": data["last_updated"] or _fmt_date(item["lastmod"]),
        "sitemap_lastmod": item["lastmod"],
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = OUT_DIR / slugify(topic) / f"{slug_from_url(item['url'])}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n\n"
    content += f"# {data['title']}\n\n{body}\n"
    path.write_text(content, encoding="utf-8")
    return path


async def worker(context, queue: asyncio.Queue, manifest: dict, failed: list, bar, delay: float, retries: int):
    page = await context.new_page()
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        for attempt in range(1, retries + 1):
            try:
                data = await extract(page, item["url"])
                if not data["body_html"]:
                    raise ValueError("empty body")
                path = write_markdown(item, data)
                rel = path.relative_to(ROOT).as_posix()
                old = manifest.get(item["url"])
                if old and old["file"] != rel:
                    (ROOT / old["file"]).unlink(missing_ok=True)
                manifest[item["url"]] = {"lastmod": item["lastmod"], "file": rel}
                break
            except Exception as e:
                tqdm.write(f"[{attempt}/{retries}] {item['url']}: {type(e).__name__}: {str(e)[:120]}")
                if attempt == retries or isinstance(e, LoginRequired):
                    failed.append({"url": item["url"], "error": f"{type(e).__name__}: {e}"[:300]})
                    break
                else:
                    await asyncio.sleep(3 * attempt)
        bar.update(1)
        await asyncio.sleep(delay + random.uniform(0, delay))
        queue.task_done()
    await page.close()


async def run(limit: int | None, force: bool, concurrency: int, delay: float, retries: int):
    items = fetch_article_urls()
    manifest = load_manifest()
    todo = [
        i for i in items
        if force or i["url"] not in manifest or manifest[i["url"]]["lastmod"] != i["lastmod"]
        or not (ROOT / manifest[i["url"]]["file"]).exists()
    ]
    if limit:
        todo = todo[:limit]
    print(f"{len(items)} articles in sitemap, {len(todo)} to scrape")
    failed: list = []
    queue: asyncio.Queue = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)
    for _ in range(concurrency):
        queue.put_nowait(None)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(locale="en-US")
        await context.route(
            re.compile(r".*\.(png|jpe?g|gif|svg|woff2?|ttf|mp4)(\?.*)?$"), lambda route: route.abort()
        )
        with tqdm(total=len(todo)) as bar:
            workers = [worker(context, queue, manifest, failed, bar, delay, retries) for _ in range(concurrency)]
            saver = asyncio.create_task(_autosave(manifest))
            await asyncio.gather(*workers)
            saver.cancel()
        await browser.close()
    save_manifest(manifest)
    FAILED.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    print(f"done: {len(todo) - len(failed)} ok, {len(failed)} failed (see {FAILED.relative_to(ROOT)})")


async def _autosave(manifest: dict):
    while True:
        await asyncio.sleep(30)
        save_manifest(manifest)


def main():
    ap = argparse.ArgumentParser(description="Scrape AskMH FAQ articles into markdown")
    ap.add_argument("--limit", type=int, help="scrape only N articles (testing)")
    ap.add_argument("--force", action="store_true", help="re-scrape even if lastmod unchanged")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--delay", type=float, default=1.0, help="base seconds between requests per worker")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(run(args.limit, args.force, args.concurrency, args.delay, args.retries))


if __name__ == "__main__":
    main()
