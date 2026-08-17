#!/usr/bin/env python3
"""Build HTML book and EPUB from chapter markdown files.

Demonstrates: assembling a multi-chapter business guide with shared styling.
Expected behavior: writes dist/book.html, dist/computer-vision-business-guide.epub,
and dist/html-bundle.zip with all assets.
"""

from __future__ import annotations

import html
import re
import shutil
import zipfile
from datetime import date
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parents[1]
BOOK_DIR = TOPIC_DIR / "book"
CHAPTERS_DIR = BOOK_DIR / "chapters"
DIST_DIR = TOPIC_DIR / "dist"
ILLUSTRATIONS = TOPIC_DIR / "assets" / "illustrations"
CSS_FILE = BOOK_DIR / "styles" / "book.css"

BOOK_TITLE = "Компьютерное зрение: вводное руководство для бизнеса"
BOOK_SUBTITLE = "Понятное руководство для сотрудников без технического бэкграунда"


def _read_chapters() -> list[tuple[str, str, str]]:
    """Return list of (filename, title, html_body)."""
    chapters: list[tuple[str, str, str]] = []
    for path in sorted(CHAPTERS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        body = _md_to_html(raw)
        chapters.append((path.stem, title, body))
    return chapters


def _md_to_html(text: str) -> str:
    """Minimal markdown to HTML converter for book chapters."""
    lines = text.splitlines()
    out: list[str] = []
    in_ul = False
    in_ol = False
    in_blockquote = False
    in_table = False
    table_rows: list[str] = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def close_blockquote() -> None:
        nonlocal in_blockquote
        if in_blockquote:
            out.append("</blockquote>")
            in_blockquote = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        out.append('<table class="data-table">')
        for i, row in enumerate(table_rows):
            cells = [c.strip() for c in row.strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
        out.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        if line.startswith("# "):
            continue  # title handled separately
        if line.startswith("|") and "|" in line[1:]:
            close_lists()
            close_blockquote()
            if re.match(r"^\|[-:\s|]+\|$", line):
                continue
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(line)
            continue
        flush_table()

        if line.startswith("> "):
            close_lists()
            if not in_blockquote:
                out.append('<blockquote class="callout">')
                in_blockquote = True
            out.append(f"<p>{_inline(line[2:])}</p>")
            continue
        close_blockquote()

        if line.startswith("- "):
            if not in_ul:
                close_lists()
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue

        m_ol = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m_ol:
            if not in_ol:
                close_lists()
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_inline(m_ol.group(2))}</li>")
            continue

        close_lists()

        if line.startswith("## "):
            out.append(f'<h2 id="{_slug(line[3:])}">{_inline(line[3:])}</h2>')
        elif line.startswith("### "):
            out.append(f'<h3 id="{_slug(line[4:])}">{_inline(line[4:])}</h3>')
        elif line.startswith("#### "):
            out.append(f'<h4>{_inline(line[5:])}</h4>')
        elif line.strip() == "":
            continue
        elif line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if m:
                alt, src = m.group(1), m.group(2)
                out.append(f'<figure class="illustration"><img src="{src}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        else:
            out.append(f"<p>{_inline(line)}</p>")

    close_lists()
    close_blockquote()
    flush_table()
    return "\n".join(out)


def _slug(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", s)[:60]


def _inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', t)
    return t


def _build_toc(chapters: list[tuple[str, str, str]]) -> str:
    items = []
    for i, (_, title, _) in enumerate(chapters, 1):
        sid = _slug(title)
        items.append(f'<li><a href="#chapter-{i}"><span class="toc-num">{i}</span>{html.escape(title)}</a></li>')
    return "\n".join(items)


def build_html(chapters: list[tuple[str, str, str]]) -> str:
    css = CSS_FILE.read_text(encoding="utf-8")
    toc = _build_toc(chapters)
    chapter_html = []
    for i, (_, title, body) in enumerate(chapters, 1):
        chapter_html.append(
            f'<section class="chapter" id="chapter-{i}">'
            f'<header class="chapter-header"><span class="chapter-num">Глава {i}</span>'
            f"<h1>{html.escape(title)}</h1></header>"
            f'<div class="chapter-body">{body}</div>'
            f'<footer class="chapter-footer"><span>{BOOK_TITLE}</span><span>— {i} —</span></footer>'
            f"</section>"
        )
    cover_img = "assets/illustrations/01_cover_hero.png"
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(BOOK_TITLE)}</title>
<style>{css}</style>
</head>
<body>
<nav class="sidebar" aria-label="Содержание">
  <div class="sidebar-brand">{html.escape(BOOK_TITLE)}</div>
  <ol class="toc">{toc}</ol>
</nav>
<main class="book">
  <section class="cover-page" id="cover">
    <img class="cover-image" src="{cover_img}" alt="Обложка">
    <div class="cover-text">
      <p class="cover-label">Knowledge Book · Business Edition</p>
      <h1>{html.escape(BOOK_TITLE)}</h1>
      <p class="cover-subtitle">{html.escape(BOOK_SUBTITLE)}</p>
      <p class="cover-meta">Версия 1.0 · {date.today().isoformat()}</p>
    </div>
  </section>
  <section class="toc-page" id="contents">
    <h1>Содержание</h1>
    <ol class="toc-full">{toc}</ol>
  </section>
  {"".join(chapter_html)}
  <section class="chapter back-cover">
    <h2>О руководстве</h2>
    <p>Это руководство подготовлено в рамках проекта Knowledge Book и предназначено для сотрудников,
    руководителей и специалистов, которые хотят понять возможности компьютерного зрения без погружения
    в программирование и математику.</p>
    <p>Технические детали и углублённые материалы — в разделах
    <a href="https://github.com">knowledge-book/topics/</a> (Computer Vision, Object Detection и др.).</p>
    <p class="copyright">© Knowledge Book · {date.today().year}</p>
  </section>
</main>
</body>
</html>"""


def build_epub(chapters: list[tuple[str, str, str]], out_path: Path) -> None:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("kb-cv-business-guide-v1")
    book.set_title(BOOK_TITLE)
    book.set_language("ru")
    book.add_author("Knowledge Book")

    style = CSS_FILE.read_text(encoding="utf-8")
    nav_css = epub.EpubItem(uid="style", file_name="style/book.css", media_type="text/css", content=style)
    book.add_item(nav_css)

    spine = ["nav"]
    toc = []

    # Cover
    cover_path = ILLUSTRATIONS / "01_cover_hero.png"
    if cover_path.exists():
        with open(cover_path, "rb") as f:
            book.set_cover("cover.png", f.read())

    for i, (_, title, body) in enumerate(chapters, 1):
        c = epub.EpubHtml(title=title, file_name=f"chapter_{i:02d}.xhtml", lang="ru")
        c.content = (
            "<html xmlns=\"http://www.w3.org/1999/xhtml\">"
            "<head>"
            f"<title>{html.escape(title)}</title>"
            "<link rel=\"stylesheet\" href=\"style/book.css\"/>"
            "</head><body>"
            "<section class=\"chapter\">"
            "<header class=\"chapter-header\">"
            f"<span class=\"chapter-num\">Глава {i}</span>"
            f"<h1>{html.escape(title)}</h1>"
            "</header>"
            f"<div class=\"chapter-body\">{body}</div>"
            "</section>"
            "</body></html>"
        )
        c.add_item(nav_css)
        book.add_item(c)
        spine.append(c)
        toc.append(epub.Link(f"chapter_{i:02d}.xhtml", title, f"ch{i}"))

    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    epub.write_epub(str(out_path), book)


def package_html_bundle(html_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(html_path, "index.html")
        css_rel = "assets/styles/book.css"
        zf.write(CSS_FILE, css_rel)
        if ILLUSTRATIONS.exists():
            for img in sorted(ILLUSTRATIONS.glob("*.png")):
                zf.write(img, f"assets/illustrations/{img.name}")


def main() -> None:
    chapters = _read_chapters()
    if len(chapters) < 10:
        raise SystemExit(f"Expected at least 10 chapters, got {len(chapters)}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    html_content = build_html(chapters)
    html_path = DIST_DIR / "index.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Copy assets next to HTML for standalone bundle
    assets_dst = DIST_DIR / "assets"
    if assets_dst.exists():
        shutil.rmtree(assets_dst)
    (assets_dst / "illustrations").mkdir(parents=True)
    (assets_dst / "styles").mkdir(parents=True)
    shutil.copy(CSS_FILE, assets_dst / "styles" / "book.css")
    if ILLUSTRATIONS.exists():
        for img in ILLUSTRATIONS.glob("*.png"):
            shutil.copy(img, assets_dst / "illustrations" / img.name)

    epub_path = DIST_DIR / "computer-vision-business-guide.epub"
    build_epub(chapters, epub_path)

    zip_path = DIST_DIR / "computer-vision-business-guide-html.zip"
    package_html_bundle(html_path, zip_path)

    page_estimate = sum(len(body.split("<p>")) + len(body.split("<h2")) for _, _, body in chapters)
    print(f"Built {len(chapters)} chapters → {html_path}")
    print(f"EPUB: {epub_path}")
    print(f"HTML bundle: {zip_path}")
    print(f"Estimated content blocks (proxy for pages): {page_estimate}")


if __name__ == "__main__":
    main()
