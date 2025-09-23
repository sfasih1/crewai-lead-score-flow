from __future__ import annotations
from pathlib import Path
from typing import List
import json
from lead_score_flow.lead_types import Persona
import re

def load_personas_from_json(path: str | Path) -> List[Persona]:
    """Load personas from a JSON file (array) or a folder of *.json files.

    JSON schema per persona object (minimal):
    {
      "id": "cio",
      "name": "Chief Information Officer",
      "roles": ["CIO", "IT Executive"],
      "industries": ["healthcare"],
      "segment": "enterprise"
    }
    Any missing optional keys are defaulted by the Persona model.
    """
    p = Path(path)
    personas: List[Persona] = []
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                try:
                    personas.append(Persona(**obj))
                except Exception as e:
                    print(f"Skipping invalid persona entry: {e}")
        elif isinstance(data, dict):
            # Two possibilities: a persona dict OR markdown-export dict with 'content'
            if "content" in data and "filename" in data:
                extracted = _persona_from_markdown_export_dict(data)
                if extracted:
                    personas.append(extracted)
                else:
                    print(f"Could not parse markdown-export persona in {p.name}")
            else:
                try:
                    personas.append(Persona(**data))
                except Exception as e:
                    print(f"File {p.name} does not conform to persona schema: {e}")
        else:
            print("Top-level JSON must be an array or a single persona object.")
    elif p.is_dir():
        for jf in p.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for obj in data:
                        try:
                            personas.append(Persona(**obj))
                        except Exception as e:
                            print(f"Skipping invalid persona in {jf.name}: {e}")
                elif isinstance(data, dict):
                    if "content" in data and "filename" in data:
                        extracted = _persona_from_markdown_export_dict(data)
                        if extracted:
                            personas.append(extracted)
                        else:
                            print(f"Could not parse markdown-export persona in {jf.name}")
                    else:
                        try:
                            personas.append(Persona(**data))
                        except Exception as e:
                            print(f"Skipping invalid persona in {jf.name}: {e}")
            except Exception as e:
                print(f"Failed reading {jf}: {e}")
    else:
        print(f"Path not found: {p}")
    return personas


def _persona_from_markdown_export_dict(d: dict) -> Persona | None:
    """Attempt to build a Persona from a dict produced by a markdown export wrapper:
    {
      "filename": "INOV Provider - Quality Coordinator ... .md",
      "content": "# Provider - Quality Coordinator\n..."
    }
    We try to infer id, name, roles, segment.
    """
    content = d.get("content", "")
    filename = d.get("filename", "persona")
    if not content:
        return None

    # Name extraction: look for first markdown heading after '#'
    m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    name = m.group(1).strip() if m else filename.split(".")[0]

    # Roles: collect bullet list items under 'Alternate Job Titles' or similar
    roles_section = re.search(r"Alternate Job Titles:\n\n(- .+?)(\n\n|$)", content, re.DOTALL)
    roles: list[str] = []
    if roles_section:
        for line in roles_section.group(1).splitlines():
            line = line.strip()
            if line.startswith("- "):
                roles.append(line[2:].strip())

    # Segment / Business Unit
    seg_m = re.search(r"Business Unit:\s*(.+)", content)
    segment = seg_m.group(1).strip() if seg_m else None

    # ID heuristic: slugify name
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower() or "persona"

    try:
        return Persona(id=slug, name=name, roles=roles, segment=segment)
    except Exception:
        return None
