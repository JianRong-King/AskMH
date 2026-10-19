import re

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

BASE = "https://www.askmh.malaysiaairlines.com"


def _cell_text(cell: Tag) -> str:
    text = " ".join(cell.get_text(" ", strip=True).split())
    return text.replace("|", "\\|")


def table_to_markdown(table: Tag) -> str:
    grid: list[list[str | None]] = []
    for r, tr in enumerate(table.find_all("tr")):
        while len(grid) <= r:
            grid.append([])
        c = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while c < len(grid[r]) and grid[r][c] is not None:
                c += 1
            text = _cell_text(cell)
            rowspan = int(cell.get("rowspan", 1) or 1)
            colspan = int(cell.get("colspan", 1) or 1)
            for dr in range(rowspan):
                while len(grid) <= r + dr:
                    grid.append([])
                row = grid[r + dr]
                for dc in range(colspan):
                    while len(row) <= c + dc:
                        row.append(None)
                    row[c + dc] = text
            c += colspan
    grid = [row for row in grid if any(row)]
    if not grid:
        return ""
    width = max(len(row) for row in grid)
    rows = [[cell or "" for cell in row] + [""] * (width - len(row)) for row in grid]
    captions = []
    while len(rows) > 1 and width > 2 and len(set(rows[0][1:])) == 1:
        label, *rest = rows.pop(0)
        captions.append(f"**{label}**" if label == rest[0] else f"**{label}: {rest[0]}**")
    for row in rows[1:]:
        if width > 1 and len(set(row)) == 1:
            row[1:] = [""] * (width - 1)
    lines = captions + [""] if captions else []
    lines += ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("/"):
            a["href"] = BASE + a["href"]
    tables = {}
    for i, table in enumerate(soup.find_all("table")):
        if table.find_parent("table"):
            continue
        key = f"TABLEPLACEHOLDER{i}X"
        tables[key] = table_to_markdown(table)
        table.replace_with(soup.new_string(f"\n\n{key}\n\n"))
    md = markdownify(str(soup), heading_style="ATX", bullets="-", strip=["img", "span"])
    for key, table_md in tables.items():
        md = md.replace(key, table_md)
    md = md.replace("\xa0", " ")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
