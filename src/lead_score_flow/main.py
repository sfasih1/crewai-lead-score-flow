#!/usr/bin/env python
import os
import sys
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# SECURITY: read OPENAI_API_KEY from environment (e.g., via .env), do NOT hardcode secrets
from dotenv import load_dotenv
# Load .env from project root (CWD) and also from this package folder
load_dotenv()  # default: searches from CWD upward
# Prefer project-local .env next to this file to avoid stale shell env overriding
load_dotenv(dotenv_path=str((Path(__file__).parent / ".env")), override=True)
# Feature flags (env)
USE_LLM = os.getenv("USE_LLM", "1") == "1"          # controls scoring crew (not used)
USE_LLM_DEEPDIVE = os.getenv("USE_LLM_DEEPDIVE", "1") == "1"  # controls deep dive crew

_api_key = None
_api_key_source = None
for key_name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY", "PERPLEXITY_API_KEY", "OPENROUTER_API_KEY"):
    val = os.getenv(key_name)
    if val:
        # Sanitize: trim surrounding whitespace/newlines to avoid auth errors
        val = val.strip()
        if not val:
            continue
        _api_key = val
        _api_key_source = key_name
        # Ensure the sanitized value is used by downstream libraries
        os.environ[key_name] = _api_key
        # Handle OpenRouter specifics
        if key_name == "OPENROUTER_API_KEY":
            # Set both legacy and new OpenAI base URL envs for maximum compatibility
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"  # legacy
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"   # new OpenAI SDK
            os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
            # Mirror key into OPENAI_API_KEY for libraries that only read this
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = _api_key
            referrer = os.getenv("OPENROUTER_REFERRER")
            if referrer:
                os.environ["HTTP_REFERER"] = referrer
        break
# IMPORTANT: Do NOT hard-exit the script here; just disable deep dive if no key
if (USE_LLM or USE_LLM_DEEPDIVE) and not _api_key:
    print("WARNING: No LLM API key found. Deep-dive analysis will be skipped.")
    USE_LLM = False
    USE_LLM_DEEPDIVE = False

if os.getenv("LITELLM_DEBUG", "0") == "1" and _api_key:
    masked = _api_key[:3] + "..." + _api_key[-4:]
    src = _api_key_source or "(unknown env var)"
    print(f"LLM enabled. Using {_api_key_source}: {masked}")
    _llm_model = os.getenv("LLM_MODEL")
    if _llm_model:
        print(f"LLM_MODEL={_llm_model}")
    print(f"DEBUG: Using key from {_api_key_source} starting with '{_api_key[:7]}' and ending with '{_api_key[-4:]}'")
    if _api_key_source == "OPENROUTER_API_KEY":
        print(f"DEBUG: OPENAI_API_BASE={os.getenv('OPENAI_API_BASE')}")
        print(f"DEBUG: OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')}")
        print(f"DEBUG: OPENAI_API_KEY set? {'yes' if os.getenv('OPENAI_API_KEY') else 'no'}")

# Optional API sanity checks without making a network call
if os.getenv("API_SANITY_CHECK", "0") == "1" and (USE_LLM or USE_LLM_DEEPDIVE):
    print("\n[API Sanity Check]")
    if _api_key_source == "OPENAI_API_KEY":
        if not (_api_key.startswith("sk-") and len(_api_key) > 20):
            print("Warning: OPENAI_API_KEY doesn't look like a standard sk-... key.")
        else:
            print("OPENAI_API_KEY format looks OK.")
    elif _api_key_source == "AZURE_OPENAI_API_KEY" or _api_key_source == "AZURE_API_KEY":
        az_ep = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
        if not az_ep:
            print("Warning: Azure key detected but AZURE_OPENAI_ENDPOINT (or AZURE_ENDPOINT) is not set.")
        else:
            print(f"Azure endpoint set: {az_ep}")
    elif _api_key_source == "PERPLEXITY_API_KEY":
        print("Perplexity API key detected. Recommended LiteLLM model prefix: perplexity/<model>.")
        print("Example: perplexity/llama-3.1-sonar-small-128k-online (lower cost)")
    elif _api_key_source == "OPENROUTER_API_KEY":
        print("OpenRouter API key detected. LiteLLM will use the OpenRouter endpoint.")
        print("Set LLM_MODEL to the desired model, e.g., 'openrouter/google/gemini-flash-1.5'")
    else:
        print("Note: Using a non-standard key env var; ensure your provider config is correct.")
    print("Set API_SANITY_CHECK=0 to silence this.")

import asyncio
from typing import List

# REMOVE CrewAI Flow decorators/imports that were breaking execution
# from crewai.flow.flow import Flow, listen, or_, router, start  # <-- removed

# Persona ingestion (Markdown-in-PDF / .md folder)
from lead_score_flow.utils.persona_loader_mdpdf import load_personas_from_folder
from lead_score_flow.utils.persona_loader_json import load_personas_from_json

# Use only the required crew for deep dive
from lead_score_flow.crews.persona_pipeline_crew.persona_pipeline_crew import PersonaPipelineCrew

# Core types
from lead_score_flow.lead_types import (
    Candidate,
    CandidatePersonaDeepDive,
    CandidatePersonaMatches,
    Persona,
)

class PersonaMatcher:
    # Removed the @start decorator — we call run() directly
    def run(self):
        print("\n=== PersonaMatcher: START ===")
        # 1. Load Leads
        candidates = self._load_leads_from_csv()
        print(f"[OK] Leads loaded: {len(candidates)} candidate(s)")

        # 2. Load Personas
        personas = self._load_personas()
        print(f"[OK] Personas loaded: {len(personas)} persona(s)")

        # 3. Run the full pipeline
        if personas:
            print("[RUN] Starting persona pipeline (lexical match -> optional deep dive)...")
            # The pipeline is async, so we run it in an event loop
            asyncio.run(self._run_persona_pipeline(candidates, personas))
        else:
            print("[SKIP] No personas loaded. Skipping pipeline.")

        print("=== PersonaMatcher: DONE ===\n")

    def _load_leads_from_csv(self) -> List[Candidate]:
        """
        Loads candidates from a CSV file.
        Priority:
        1. LEADS_CSV environment variable.
        2. Tkinter file dialog.
        """
        import csv
        current_dir = Path(__file__).parent
        csv_file_path = os.getenv("LEADS_CSV")

        if csv_file_path and Path(csv_file_path).exists():
            print(f"[INPUT] Using LEADS_CSV from environment: {csv_file_path}")
        else:
            print("[PROMPT] Opening file dialog to select your Contacts CSV file...")
            try:
                from tkinter import Tk, filedialog
                root = Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                csv_file_path = filedialog.askopenfilename(
                    title="Select Your Contacts CSV File",
                    initialdir=str(current_dir),
                    filetypes=[["CSV Files", "*.csv"]]
                )
                root.destroy()
            except (ImportError, RuntimeError) as e:
                print("\n--- GUI ERROR ---")
                print("Could not open the graphical file dialog.")
                print(f"DETAILS: {e}")
                print("WORKAROUND: Set the 'LEADS_CSV' environment variable to the full path of your contacts file.")
                raise SystemExit(1)

        if not csv_file_path:
            print("[CANCELLED] No file was selected. Exiting.")
            raise SystemExit(1)
        
        print(f"[READ] Leads CSV: {csv_file_path}")

        candidates = []
        with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                candidate_data = {
                    "id": row.get("Id", ""),
                    "job_title": row.get("Job Title", ""),
                    "company_name": row.get("Company Name", ""),
                    "primary_line_of_business": row.get("Primary Line of Business", ""),
                    "primary_line_of_business_segment": row.get("Primary Line of Business Segment", ""),
                    "primary_line_of_business_unique_id": row.get("Primary Line of Business_Unique ID", ""),
                    "job_title_department": row.get("Job Title Department", ""),
                    "job_title_c": row.get("Job Title (C)", ""),
                    "job_function": row.get("Job Function", ""),
                    "job_function_segment": row.get("Job Function Segment", ""),
                    "job_level": row.get("Job Level", ""),
                    "job_level_segment": row.get("Job Level Segment", ""),
                    "job_segment": row.get("Job Segment", ""),
                    "role": row.get("Role", ""),
                    "role_1": row.get("Role.1", ""),
                    "secondary_line_of_business": row.get("Secondary Line of Business", ""),
                    "source_system": row.get("Source System", ""),
                }
                candidate = Candidate(**candidate_data)
                candidates.append(candidate)
        print(f"[OK] Parsed {len(candidates)} candidate row(s) from CSV")
        return candidates

    def _load_personas(self) -> List[Persona]:
        """
        Loads personas automatically from JSON files in the project root directory.
        """
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Navigate up to the project root
        print(f"[SCAN] Looking for persona JSON files in: {project_root}")

        json_files = list(project_root.glob("*.json"))
        
        if not json_files:
            print("--- ERROR --- No persona JSON files found in the project root directory.")
            print("Please ensure your persona files (e.g., '*.pdf.json') are next to 'pyproject.toml'.")
            raise SystemExit(1)

        print(f"[FOUND] {len(json_files)} JSON file(s): {[f.name for f in json_files]}")

        loaded_personas = []
        for jf in json_files:
            try:
                loaded_personas.extend(load_personas_from_json(jf))
            except Exception as e:
                print(f"[WARN] Failed to load personas from {jf.name}: {e}")

        if not loaded_personas:
            print("--- ERROR --- No personas could be loaded from the found JSON files.")
            raise SystemExit(1)

        # De-duplicate and return
        unique = { (p.id or p.name): p for p in loaded_personas }
        final_personas = list(unique.values())
        print(f"[OK] Loaded & de-duplicated {len(final_personas)} persona(s)")
        return final_personas

    async def _run_persona_pipeline(self, candidates: List[Candidate], personas: List[Persona]):
        """Runs the lightweight matching and deep dive analysis."""
        # 1. Lightweight Persona Matching
        print("\n[STEP] Lightweight Persona Matching: START")
        from lead_score_flow.lead_types import CandidatePersonaMatches, PersonaMatchItem
        import csv

        def simple_score(title: str, persona) -> float:
            t = set(w.lower() for w in title.split() if len(w) > 2)
            pname = persona.name.lower()
            roles = " ".join(persona.roles or []).lower()
            bag = set([w for w in (pname + " " + roles).split() if len(w) > 2])
            inter = t & bag
            if not t or not bag:
                return 0.0
            return round(100.0 * len(inter) / len(t), 1)

        all_matches: list[CandidatePersonaMatches] = []
        for cand in candidates:
            scores = sorted(
                [(simple_score(cand.job_title, p), p) for p in personas if simple_score(cand.job_title, p) > 0],
                reverse=True, key=lambda x: x[0]
            )
            top_items = [PersonaMatchItem(persona_id=p.id, persona_name=p.name, score=s, rationale="token_overlap") for s, p in scores[:3]]
            all_matches.append(CandidatePersonaMatches(candidate_id=cand.id, candidate_title=cand.job_title, matches=top_items))

        self._export_top3_persona_matches(all_matches)
        print("[STEP] Lightweight Persona Matching: DONE (exported persona_top3.csv)")

        # 2. Deep Dive Analysis
        if not USE_LLM_DEEPDIVE:
            print("[STEP] Deep Dive: SKIPPED (no API key or feature disabled)")
        else:
            print("[STEP] Deep Dive: START")
            await self._execute_deep_dive_logic(candidates, all_matches)
            print("[STEP] Deep Dive: DONE")

    def _export_top3_persona_matches(self, all_matches: List[CandidatePersonaMatches]):
        """Exports top 3 persona matches to a CSV file."""
        import csv
        out = Path(__file__).parent / "persona_top3.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "candidate_id", "candidate_title",
                "persona1_id", "persona1_name", "persona1_score",
                "persona2_id", "persona2_name", "persona2_score",
                "persona3_id", "persona3_name", "persona3_score",
            ]
            w.writerow(header)
            for row in all_matches:
                m = row.matches
                cells = [row.candidate_id, row.candidate_title]
                for i in range(3):
                    if i < len(m):
                        cells.extend([m[i].persona_id, m[i].persona_name, m[i].score])
                    else:
                        cells.extend(["", "", ""])
                w.writerow(cells)
        print(f"[WRITE] {out.name}")

    async def _execute_deep_dive_logic(self, candidates: List[Candidate], persona_matches: List[CandidatePersonaMatches]):
        """Executes the deep dive analysis for each candidate."""
        from lead_score_flow.lead_types import CandidatePersonaDeepDive
        import json
        import asyncio

        sc_map = {c.id: c for c in candidates}
        tasks = []

        debug_dir = Path(__file__).parent / "_debug_payloads"
        try:
            debug_dir.mkdir(exist_ok=True)
        except Exception:
            pass

        async def run_deepdive_for(cpm: CandidatePersonaMatches):
            sc = sc_map.get(cpm.candidate_id)
            if not sc:
                print(f"[WARN] Candidate not found for matches id={cpm.candidate_id}; skipping")
                return None

            personas_payload = [{"persona_id": m.persona_id, "persona_name": m.persona_name, "lexical_score": m.score} for m in cpm.matches]
            personas_payload_json = json.dumps(personas_payload, ensure_ascii=False)

            # Write a per-candidate snapshot for debugging/traceability
            try:
                snap = {
                    "candidate": {
                        "id": sc.id,
                        "job_title": sc.job_title,
                        "company_name": sc.company_name,
                        "job_function": sc.job_function,
                        "job_level": sc.job_level,
                        "line_of_business": sc.primary_line_of_business,
                    },
                    "matches": personas_payload,
                }
                (debug_dir / f"candidate_{sc.id}_payload.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[WARN] Failed to write debug payload for candidate {sc.id}: {e}")

            try:
                crew = PersonaPipelineCrew().crew()
                inputs = {
                    "candidate_id": cpm.candidate_id or "",
                    "candidate_title": sc.job_title or "",
                    "company_name": sc.company_name or "",
                    "job_function": sc.job_function or "",
                    "job_level": sc.job_level or "",
                    "line_of_business": sc.primary_line_of_business or "",
                    "persona_matches_json": personas_payload_json,
                }
                # Extra validation log before calling crew
                print(f"[DeepDive/Input] id={inputs['candidate_id']} title='{inputs['candidate_title']}' matches={len(personas_payload)}")
                result = await asyncio.wait_for(
                    crew.kickoff_async(inputs=inputs),
                    timeout=float(os.getenv("DEEPDIVE_TIMEOUT_SECONDS", "120")),
                )
                if hasattr(result, "pydantic") and isinstance(result.pydantic, CandidatePersonaDeepDive):
                    return result.pydantic
                else:
                    print(f"[WARN] Deep dive returned non-pydantic result for candidate {cpm.candidate_id}")
            except Exception as e:
                # Print full traceback to identify the source of 'NoneType.startswith' errors
                import traceback
                tb = traceback.format_exc()
                print(f"[ERROR] Deep dive failed for candidate {cpm.candidate_id}: {e}\n{tb}")
            return None

        for cpm in persona_matches:
            if cpm.matches:
                tasks.append(asyncio.create_task(run_deepdive_for(cpm)))

        gathered = await asyncio.gather(*tasks)
        results = [r for r in gathered if r]

        if results:
            self._export_deep_dive_results(results)
            print(f"[OK] Deep dive analysis complete. Results: {len(results)} candidates")
        else:
            print("[INFO] No deep dive results were produced.")

    def _export_deep_dive_results(self, results: List[CandidatePersonaDeepDive]):
        """Exports deep dive results to CSV files."""
        import csv
        # Long form export
        out_long = Path(__file__).parent / "persona_deepdive_long.csv"
        with out_long.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "candidate_id", "candidate_title", "persona_id", "persona_name", "relevance", "recommended", "overall_fit", "strengths", "risks", "messaging_hooks"
            ])
            for r in results:
                for a in r.assessments:
                    w.writerow([
                        r.candidate_id,
                        r.candidate_title,
                        a.persona_id,
                        a.persona_name,
                        a.relevance,
                        a.recommended,
                        a.overall_fit,
                        "|".join(a.strengths or []),
                        "|".join(a.risks or []),
                        "|".join(a.messaging_hooks or []),
                    ])
        print(f"[WRITE] {out_long.name}")

        # Best persona summary (wide)
        out_best = Path(__file__).parent / "persona_deepdive_best.csv"
        with out_best.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["candidate_id", "candidate_title", "best_persona_id", "best_persona_name", "relevance", "overall_fit", "reasoning"])
            for r in results:
                best = next((a for a in r.assessments if a.recommended), None)
                if best:
                    w.writerow([
                        r.candidate_id,
                        r.candidate_title,
                        best.persona_id,
                        best.persona_name,
                        best.relevance,
                        best.overall_fit,
                        r.reasoning.replace('\n', ' '),
                    ])
        print(f"[WRITE] {out_best.name}")


def kickoff():
    """Run the flow."""
    matcher = PersonaMatcher()
    matcher.run()


def plot():
    """Plot the flow."""
    print("The application has been simplified and no longer uses a complex flow diagram.")


if __name__ == "__main__":
    kickoff()
