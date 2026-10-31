"""a rough local page of a staged repo: its card as the hub would show it, its tags and files.

it covers the markdown a card uses (headings, paragraphs, lists, tables, code blocks,
images, links), since the hub's own renderer is not available offline."""

import html
import re
from pathlib import Path

GIB, MIB = 1 << 30, 1 << 20
STYLE = """
body { margin: 0; background: #fff; color: #1f2937; font: 15px/1.6 system-ui, sans-serif; }
header { border-bottom: 1px solid #e5e7eb; padding: 16px 32px; background: #f9fafb; }
h1.repo { font-size: 20px; margin: 0 0 8px; }
.tag { display: inline-block; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0 8px; margin: 2px 4px 2px 0;
       font-size: 13px; background: #fff; }
main { display: flex; gap: 32px; max-width: 1200px; margin: 0 auto; padding: 24px 32px; }
article { flex: 1; min-width: 0; }
aside { width: 300px; font-size: 13px; }
img { max-width: 100%; border-radius: 8px; }
table { border-collapse: collapse; display: block; overflow-x: auto; font-size: 13px; }
th, td { border: 1px solid #e5e7eb; padding: 4px 8px; text-align: left; }
pre { background: #f3f4f6; border-radius: 8px; padding: 12px; overflow-x: auto; font-size: 13px; }
code { background: #f3f4f6; border-radius: 4px; padding: 0 3px; font-size: 13px; }
pre code { background: none; padding: 0; }
h1, h2 { border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
"""


def inline(text):
    """a line's images, links and code, with everything else escaped."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', out)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)


def table(rows):
    """a markdown table's lines, its second line being the rule under the header."""
    cells = [[c.strip() for c in row.strip("|").split("|")] for row in rows]
    head = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
    body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in cells[2:])
    return f"<table><tr>{head}</tr>{body}</table>"


def blocks(lines):
    """split a card's lines at blank lines and code fences: (kind, lines)."""
    out, para, fenced = [], [], None
    for line in lines:
        if fenced is not None:
            if line.startswith("```"):
                out.append(("code", fenced))
                fenced = None
            else:
                fenced.append(line)
        elif line.startswith("```"):
            out.append(("text", para))
            para, fenced = [], []
        elif line.strip():
            para.append(line)
        else:
            out.append(("text", para))
            para = []
    return [*out, ("text", para)]


def text_block(lines):
    """a heading, table, list, paragraph, or html the hub passes through (a collapsed table)."""
    if lines[0].startswith("<"):
        return "\n".join(lines)
    if lines[0].startswith("#"):
        level = len(lines[0]) - len(lines[0].lstrip("#"))
        return f"<h{level}>{inline(lines[0].lstrip('# '))}</h{level}>"
    if lines[0].startswith("|"):
        return table(lines)
    if lines[0].startswith("- "):
        items = re.split(r"\n- ", "\n" + "\n".join(lines))[1:]
        return "<ul>" + "".join(f"<li>{inline(' '.join(i.split()))}</li>" for i in items) + "</ul>"
    return f"<p>{inline(' '.join(lines))}</p>"


def render(markdown):
    """a card's body as html."""
    parts = []
    for kind, lines in blocks(markdown.splitlines()):
        if kind == "code":
            parts.append("<pre><code>" + html.escape("\n".join(lines), quote=False) + "</code></pre>")
        elif lines:
            parts.append(text_block(lines))
    return "\n".join(parts)


def split_card(card):
    """(the front matter's {key: value or list}, the card's body)."""
    _, front, body = card.split("---\n", 2)
    meta, key = {}, None
    for line in front.splitlines():
        if line.startswith("- "):
            meta[key].append(line[2:])
        else:
            key, _, value = line.partition(": ")
            key = key.rstrip(":")
            meta[key] = value or []
    return meta, body


def size(path):
    n = path.stat().st_size
    return f"{n / GIB:.1f} GiB" if n >= GIB else f"{n / MIB:.1f} MiB"


def license_of(meta):
    return meta.get("license_name") or meta.get("license", "")


def page(repo, stage, link):
    """the page of a staged directory; link is that directory as the page's files see it."""
    meta, body = split_card((stage / "README.md").read_text())
    chips = [license_of(meta), f"adapter of {meta.get('base_model', '')}", *meta.get("tags", [])]
    files = "".join(f"<tr><td>{html.escape(f.name)}</td><td>{size(f)}</td></tr>" for f in sorted(stage.iterdir()))
    rendered = re.sub(r'src="(?!https?:)([^"]+)"', lambda m: f'src="{link}/{m[1]}"', render(body))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(repo)}</title><style>{STYLE}</style></head><body>
<header><h1 class="repo">{html.escape(repo)}</h1>{"".join(f'<span class="tag">{html.escape(c)}</span>' for c in chips)}
</header>
<main><article>{rendered}</article><aside><h3>files</h3><table>{files}</table></aside></main>
</body></html>
"""


def write_preview(repo, stage):
    """the page beside the staging directory, so that an upload of it leaves the page out."""
    path = Path(f"{stage}.preview.html")
    path.write_text(page(repo, stage, stage.name))
    return path
