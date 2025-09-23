# src/lead_score_flow/utils/persona_loader_mdpdf.py
from __future__ import annotations
import re, pathlib
from typing import List, Tuple, Optional

from pypdf import PdfReader

try:
    import yaml
except Exception:  # yaml is optional; we guard it
    yaml = None

from lead_score_flow.lead_types import Persona

_slug_re = re.compile(r"[^a-z0-9]+", re.I)
def _slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = _slug_re.sub("-", s).strip("-")
    return s or "persona"

def _read_text_from_pdf(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def _read_text_from_md(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _extract_frontmatter(text: str) -> dict:
    """
    If the PDF preserves a YAML front matter block (--- ... ---) from Markdown,
    we parse it. Otherwise return {}.
    """
    if yaml is None:
        return {}
    m = re.search(r"(?s)^---\s*(.*?)\s*---\s*", text)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1)) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _find_one(text: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None

def _extract_section_block(text: str, section_names: List[str]) -> Optional[str]:
    """
    Extract a block of text following a section label (e.g., 'Titles:' or 'Roles:')
    until a blank line or the next heading-ish marker.
    """
    # Allow 'Section:\n...' or just 'Section ...'
    for name in section_names:
        # Match 'Section:' then capture lines until a heading/blank
        pat = rf"(?is)(?:^|\n){re.escape(name)}\s*:?\s*(.+?)(?:\n\s*\n|(?:\n#)|\Z)"
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None

def _split_roles(s: str) -> List[str]:
    if not s:
        return []
    s = s.replace("_x000D_", " ").replace("\r", " ").replace("\n", " ")
    # Split by comma or bullet markers
    parts = re.split(r"[,\u2022•\-–]+", s)
    out, seen = [], set()
    for p in parts:
        q = p.strip(" .")
        if not q:
            continue
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out

def _first_heading_as_name(text: str) -> Optional[str]:
    # H1 like '# Persona: Name' or '# Name'
    m = re.search(r"(?m)^\s*#\s*(?:Persona\s*:\s*)?(.+?)\s*$", text)
    if m:
        return m.group(1).strip()
    # Or 'Persona Name: ...'
    m = re.search(r"(?im)^\s*Persona(?:\s*Name)?\s*:\s*(.+?)\s*$", text)
    if m:
        return m.group(1).strip()
    return None

def _parse_persona_text(text: str, fallback_name: str, src_path: pathlib.Path) -> Persona:
    meta = _extract_frontmatter(text)  # {} if not present or yaml missing

    # NAME
    name = meta.get("name") if isinstance(meta, dict) else None
    if not name:
        name = _first_heading_as_name(text) or fallback_name

    # SEGMENT / INDUSTRY (we accept any of these labels)
    segment = (
        meta.get("segment")
        or meta.get("business_unit")
        or meta.get("industry")
        or _find_one(text, [
            r"(?im)^\s*Business\s*Unit\s*:\s*(.+)$",
            r"(?im)^\s*Segment\s*:\s*(.+)$",
            r"(?im)^\s*Industry\s*:\s*(.+)$",
        ])
        or ""
    ).strip()

    # ROLES (Titles)
    roles_src = (
        meta.get("roles")
        or meta.get("titles")
        or _extract_section_block(text, ["Titles", "Roles"])
        or _find_one(text, [r"(?im)^\s*Titles\s*:\s*(.+)$", r"(?im)^\s*Roles\s*:\s*(.+)$"])
        or ""
    )
    # meta roles may already be a list
    if isinstance(roles_src, list):
        roles = [str(x).strip() for x in roles_src if str(x).strip()]
    else:
        roles = _split_roles(str(roles_src))

    # DESCRIPTION (optional)
    desc = (
        meta.get("description")
        or _extract_section_block(text, ["Description", "Overview", "About"])
        or _find_one(text, [r"(?im)^\s*Description\s*:\s*(.+)$", r"(?im)^\s*Overview\s*:\s*(.+)$"])
        or ""
    )

    # LINK (optional, fallback to file path as a local "link")
    link = (
        meta.get("link")
        or _find_one(text, [r"(?im)^\s*Link\s*:\s*(.+)$"])
        or str(src_path)
    )

    pid = _slugify(f"{segment}-{name}") if segment else _slugify(name)
    persona = Persona(
        id=pid,
        name=name,
        segment=segment or None,
        roles=roles,
        description=desc,
        link=link,
    )
    if persona.segment and not persona.industries:
        persona.industries = [persona.segment]
    return persona

def load_personas_from_folder(folder: str | pathlib.Path) -> List[Persona]:
    p = pathlib.Path(folder).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Persona folder not found: {p}")

    personas: List[Persona] = []
    files = list(p.glob("*.pdf")) + list(p.glob("*.PDF")) + list(p.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No persona PDFs or MD files found in: {p}")

    for f in sorted(files):
        try:
            if f.suffix.lower() == ".pdf":
                text = _read_text_from_pdf(f)
            else:
                text = _read_text_from_md(f)
            # fallback name = filename sans extension
            persona = _parse_persona_text(text, f.stem, f)
            personas.append(persona)
        except Exception as e:
            print(f"[WARN] Failed to parse persona file '{f.name}': {e}")
            continue
    return personas
