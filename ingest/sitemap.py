import hashlib
import re
import xml.etree.ElementTree as ET

import httpx

BASE = "https://www.askmh.malaysiaairlines.com"
SITEMAP_INDEX = f"{BASE}/faq/s/sitemap.xml"
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
HEADERS = {"User-Agent": "Mozilla/5.0 (askmh-chatbot research crawler)"}


def _get(client: httpx.Client, url: str) -> ET.Element:
    r = client.get(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return ET.fromstring(r.content)


def fetch_article_urls() -> list[dict]:
    with httpx.Client(headers=HEADERS) as client:
        index = _get(client, SITEMAP_INDEX)
        children = [loc.text.strip() for loc in index.findall("sm:sitemap/sm:loc", NS)]
        articles: dict[str, dict] = {}
        for child in children:
            for node in _get(client, child).findall("sm:url", NS):
                loc = node.findtext("sm:loc", default="", namespaces=NS).strip()
                if "/s/article/" not in loc:
                    continue
                lastmod = node.findtext("sm:lastmod", default="", namespaces=NS).strip()
                prev = articles.get(loc)
                if prev is None or lastmod > prev["lastmod"]:
                    articles[loc] = {"url": loc, "lastmod": lastmod}
    return sorted(articles.values(), key=lambda a: a["url"])


def slug_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", slug).strip("-") or "article"
    if len(slug) > 120:
        slug = slug[:110].rstrip("-") + "-" + hashlib.sha1(url.encode()).hexdigest()[:8]
    return slug


if __name__ == "__main__":
    urls = fetch_article_urls()
    print(f"{len(urls)} articles")
    for a in urls[:5]:
        print(a)
