#!/usr/bin/env python
import os
# SECURITY: read OPENAI_API_KEY from environment (e.g., via .env), do NOT hardcode secrets
from dotenv import load_dotenv
from pathlib import Path as _Path
# Load .env from project root (CWD) and also from this package folder
load_dotenv()  # default: searches from CWD upward
# Prefer project-local .env next to this file to avoid stale shell env overriding
load_dotenv(dotenv_path=str((_Path(__file__).parent / ".env")), override=True)
# Feature flags (env)
USE_LLM = os.getenv("USE_LLM", "1") == "1"          # controls scoring crew
USE_LLM_DEEPDIVE = os.getenv("USE_LLM_DEEPDIVE", "1") == "1"  # controls deep dive crew

_api_key = None
_api_key_source = None
for key_name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY", "PERPLEXITY_API_KEY"):
    val = os.getenv(key_name)
    if val:
        _api_key = val
        _api_key_source = key_name
        break
if (USE_LLM or USE_LLM_DEEPDIVE) and not _api_key:
    print("ERROR: No API key found. Set OPENAI_API_KEY (or AZURE_OPENAI_API_KEY / AZURE_API_KEY) in your environment or .env file.")
    print("Tip: create a .env file alongside main.py with a line: OPENAI_API_KEY=sk-...  (restart your run after saving)")
    print("Alternatively in PowerShell:  $env:OPENAI_API_KEY=\"sk-...\"  and re-run.")
    raise SystemExit(1)
if os.getenv("LITELLM_DEBUG", "0") == "1" and _api_key:
    masked = _api_key[:3] + "..." + _api_key[-4:]
    src = _api_key_source or "(unknown env var)"
    print(f"LLM enabled. Using {_api_key_source}: {masked}")
if os.getenv("LITELLM_DEBUG") == "1":
    try:
        import litellm
        litellm._turn_on_debug()
        print("LiteLLM debug enabled (LITELLM_DEBUG=1)")
    except Exception:
        pass

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
    else:
        print("Note: Using a non-standard key env var; ensure your provider config is correct.")
    print("Set API_SANITY_CHECK=0 to silence this.")

import asyncio
from typing import List
from pathlib import Path

from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lead_score_flow.lead_types import CandidatePersonaMatches

# Persona ingestion (Markdown-in-PDF / .md folder)
from lead_score_flow.utils.persona_loader_mdpdf import load_personas_from_folder
from lead_score_flow.utils.persona_loader_json import load_personas_from_json

# Persona matching utilities
# Removed best persona single-match usage per new requirements


# Your crews and constants
from lead_score_flow.constants import JOB_DESCRIPTION
from lead_score_flow.crews.lead_response_crew.lead_response_crew import LeadResponseCrew
from lead_score_flow.crews.lead_score_crew.lead_score_crew import LeadScoreCrew
from lead_score_flow.crews.persona_pipeline_crew.persona_pipeline_crew import PersonaPipelineCrew

# Core types (use your existing module where these Pydantic models live)
from lead_score_flow.lead_types import (
    Candidate,
    CandidateScore,
    LeadPersonaMatch,
    Persona,
    ScoredCandidate,
    CandidatePersonaMatches,
    CandidatePersonaDeepDive,
)

from lead_score_flow.utils.candidateUtils import combine_candidates_with_scores


class LeadScoreState(BaseModel):
    candidates: List[Candidate] = []
    candidate_score: List[CandidateScore] = []
    hydrated_candidates: List[ScoredCandidate] = []
    scored_leads_feedback: str = ""

    personas: List[Persona] = []
    persona_folder: str = ""
    # NEW: multi-match lightweight results
    persona_multi_matches: List[CandidatePersonaMatches] = []
    # Deep dive outputs
    persona_deepdive_results: List["CandidatePersonaDeepDive"] = []
    # Idempotency guards
    persona_match_ran: bool = False
    deep_dive_ran: bool = False
    persona_match_in_progress: bool = False
    deep_dive_in_progress: bool = False

    # NEW: best persona per lead (computed after scoring)
    # Removed: single best persona matches no longer needed
    lead_persona_matches: List[LeadPersonaMatch] = []  # kept for backward compatibility (unused)


class LeadScoreFlow(Flow[LeadScoreState]):
    initial_state = LeadScoreState

    @start()
    def load_leads(self):
        import csv
        from tkinter import Tk, filedialog

        current_dir = Path(__file__).parent

        # --- Auto-load JSON personas (search order):
        # 1. PERSONA_DIR env var (if set)
        # 2. crews/persona_match_crew
        # 3. project root (all *.json / *pdf.json)
        loaded_auto: list[Persona] = []

        def _load_many(files):
            acc = []
            for jf in files:
                try:
                    acc.extend(load_personas_from_json(jf))
                except Exception as e:
                    print(f"Failed to load personas from {jf}: {e}")
            return acc

        env_dir = os.getenv("PERSONA_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                candidates = []
                if p.is_file() and p.suffix.lower() == ".json":
                    candidates = [p]
                elif p.is_dir():
                    candidates = list(p.glob("*.json"))
                loaded_auto = _load_many(candidates)
                if loaded_auto:
                    self.state.personas = loaded_auto
                    self.state.persona_folder = str(p)
                    print(f"Auto-loaded {len(loaded_auto)} persona(s) from PERSONA_DIR={p}")

        if not self.state.personas:
            crew_dir = current_dir / "crews" / "persona_match_crew"
            if crew_dir.exists():
                json_files = list(crew_dir.glob("*.json"))
                if json_files:
                    loaded_auto = _load_many(json_files)
                    if loaded_auto:
                        self.state.personas = loaded_auto
                        self.state.persona_folder = str(crew_dir)
                        print(f"Auto-loaded {len(loaded_auto)} persona(s) from {crew_dir}")

        if not self.state.personas:
            # search project root (where script launched) for persona JSON exports
            project_root = Path.cwd()
            root_json = list(project_root.glob("*.json"))
            # Filter likely persona files (contain 'INOV' or have 'Provider'/'Payer' etc.)
            persona_like = [f for f in root_json if any(k in f.name for k in ["INOV", "Provider", "Payer", "Pharmacy", "Insights"]) ] or root_json
            if persona_like:
                loaded_auto = _load_many(persona_like)
                if loaded_auto:
                    self.state.personas = loaded_auto
                    self.state.persona_folder = str(project_root)
                    print(f"Auto-loaded {len(loaded_auto)} persona(s) from project root")

        # De-duplicate personas by (id or name)
        if self.state.personas:
            unique = {}
            for per in self.state.personas:
                key = per.id or per.name
                if key not in unique:
                    unique[key] = per
            if len(unique) != len(self.state.personas):
                print(f"De-duplicated personas: {len(self.state.personas)} -> {len(unique)}")
            self.state.personas = list(unique.values())

        # If not auto-loaded, fall back to interactive selection
        if not self.state.personas:
            # --- Tkinter dialog for persona folder selection ---
            print("\nSelect your PERSONA folder (PDF/MD) OR a JSON file/folder using the dialog.")
            root = Tk()
            root.withdraw()  # Hide the main window
            dpath = filedialog.askdirectory(title="Select Persona Folder", initialdir=str(current_dir))
            root.destroy()
            if not dpath:
                dpath = str(current_dir / "personas")
                print(f"No folder selected. Using default: {dpath}")
            else:
                print(f"Selected persona folder: {dpath}")
            self.state.persona_folder = dpath

            # Try JSON first (if chosen path includes .json or contains json files), else fallback
            loaded_personas = []
            try:
                p = Path(dpath)
                if p.is_file() and p.suffix.lower() == ".json":
                    loaded_personas = load_personas_from_json(p)
                elif p.is_dir() and list(p.glob("*.json")):
                    loaded_personas = load_personas_from_json(p)
            except Exception as e:
                print(f"JSON persona load attempt failed: {e}")
            if not loaded_personas:
                try:
                    loaded_personas = load_personas_from_folder(dpath)
                except Exception as e:
                    print(f"WARNING: Could not load personas from '{dpath}': {e}")
            self.state.personas = loaded_personas or []
            print(f"Loaded {len(self.state.personas)} persona(s) from: {dpath}")
        else:
            print(f"Using auto-loaded personas from: {self.state.persona_folder}")

        # Logging summary
        if self.state.personas:
            sample_names = ", ".join([p.name for p in self.state.personas[:3]])
            print(f"Persona load summary: {len(self.state.personas)} personas. First few: {sample_names}")
        else:
            print("No personas loaded (will proceed without persona matching unless added later).")

        # --- NEW: Prefer LEADS_CSV env var if provided, else ask via dialog ---
        csv_file_path = os.getenv("LEADS_CSV", "")
        if csv_file_path and Path(csv_file_path).exists():
            print(f"Using LEADS_CSV from environment: {csv_file_path}")
        else:
            print("\nSelect your LEADS CSV file using the dialog.")
            root = Tk()
            root.withdraw()  # Hide the main window
            csv_file_path = filedialog.askopenfilename(title="Select Leads CSV File", initialdir=str(current_dir), filetypes=[["CSV Files", "*.csv"]])
            root.destroy()
            if not csv_file_path:
                print("No file selected. Exiting.")
                exit()
            else:
                print(f"Selected leads file: {csv_file_path}")

        candidates = []
        with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Map all CSV columns to Candidate fields
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

        # Update the state with the loaded candidates
        self.state.candidates = candidates

    @listen(or_(load_leads, "scored_leads_feedback"))
    async def score_leads(self):
        print("Scoring leads")
        tasks = []

        async def score_single_candidate(candidate: Candidate):
            try:
                if not USE_LLM:
                    # Offline deterministic score (cheap baseline)
                    jt = (candidate.job_title or "").lower()
                    base = 10.0
                    if "director" in jt or "vp" in jt or "chief" in jt:
                        base = 35.0
                    elif "manager" in jt:
                        base = 25.0
                    from lead_score_flow.lead_types import CandidateScore
                    self.state.candidate_score.append(CandidateScore(candidate=candidate, score=base))
                    return
                result = await (
                    LeadScoreCrew()
                    .crew()
                    .kickoff_async(
                        inputs={
                            "candidate_id": candidate.id,
                            "name": candidate.job_title,
                            "bio": candidate.company_name,  # You may want to update this to a richer bio
                            "job_description": JOB_DESCRIPTION,
                            "additional_instructions": self.state.scored_leads_feedback,
                        }
                    )
                )
                self.state.candidate_score.append(result.pydantic)
            except Exception as e:
                # Log error and append a failed score for this candidate
                print(f"Error scoring candidate {candidate.id} ({candidate.job_title}): {e}")
                from lead_score_flow.lead_types import CandidateScore
                self.state.candidate_score.append(CandidateScore(candidate=candidate, score=0.0))

        for candidate in self.state.candidates:
            print("Scoring candidate:", candidate.job_title)
            task = asyncio.create_task(score_single_candidate(candidate))
            tasks.append(task)

        candidate_scores = await asyncio.gather(*tasks)
        print("Finished scoring leads: ", len(candidate_scores))

    @router(score_leads)
    def human_in_the_loop(self):
        """
        Unchanged UX for HITL, but now we also:
          - hydrate/sort leads
          - compute best persona per lead (and export CSVs)
        """
        print("Finding the top 3 candidates for human to review")

        # Hydrate: combine candidates with their scores
        self.state.hydrated_candidates = combine_candidates_with_scores(
            self.state.candidates, self.state.candidate_score
        )

        # Sort descending by score
        sorted_candidates = sorted(
            self.state.hydrated_candidates, key=lambda c: c.score, reverse=True
        )
        self.state.hydrated_candidates = sorted_candidates

        # Export raw scores
        import csv as _csv
        import os
        output_csv = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lead_scores.csv')))
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = _csv.writer(file)
            writer.writerow(["id", "job_title", "company_name", "score", "reason"])
            for sc in sorted_candidates:
                writer.writerow([
                    sc.candidate.id,
                    sc.candidate.job_title,
                    sc.candidate.company_name,
                    sc.score,
                    getattr(sc, "reason", "")
                ])
        print(f"All scored candidates saved to {output_csv}")
        # Immediately proceed to lightweight persona matching if personas exist and not yet run
        print(
            f"Router check -> personas: {bool(self.state.personas)}, "
            f"match_ran: {self.state.persona_match_ran}, "
            f"match_in_progress: {self.state.persona_match_in_progress}"
        )
        if self.state.personas and not self.state.persona_match_ran and not self.state.persona_match_in_progress:
            print("Router: scheduling lightweight persona matching (top 3 per candidate)...")
            return "generate_persona_matches"
        if not self.state.personas:
            print("No personas available; flow complete.")
            return None
        # Persona matching already executed
        print("Persona matching already completed; flow complete.")
        return None

    @listen("generate_emails")
    async def write_and_save_emails(self):
        import re

        print("Writing and saving emails for all leads.")

        # Determine the top 3 candidates to proceed with
        top_candidate_ids = {
            candidate.id for candidate in self.state.hydrated_candidates[:3]
        }

        # Map best persona per lead (if matches exist)
        best_map = {m.lead_id: m for m in (self.state.lead_persona_matches or [])}

        tasks = []

        # Create the directory 'email_responses' if it doesn't exist
        output_dir = Path(__file__).parent / "email_responses"
        print("output_dir:", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        async def write_email(candidate):
            # Check if the candidate is among the top 3
            proceed_with_candidate = candidate.id in top_candidate_ids

            # Fetch persona match (may be absent if no personas loaded)
            pm = best_map.get(candidate.id)

            # Kick off the LeadResponseCrew for each candidate
            result = await (
                LeadResponseCrew()
                .crew()
                .kickoff_async(
                    inputs={
                        "candidate_id": candidate.id,
                        "name": candidate.job_title,
                        "bio": candidate.company_name,  # You may want to update this to a richer bio
                        "proceed_with_candidate": proceed_with_candidate,

                        # NEW: persona context for email personalization
                        "persona_id": getattr(pm, "best_persona_id", ""),
                        "persona_name": getattr(pm, "best_persona_name", ""),
                        "persona_similarity": getattr(pm, "similarity_score", 0.0),
                        "persona_alignment": getattr(pm, "alignment_score", 0.0),
                    }
                )
            )

            # Sanitize the candidate's name to create a valid filename
            safe_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", candidate.job_title)
            filename = f"{safe_name}.txt"
            print("Filename:", filename)

            # Write the email content to a text file
            file_path = output_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.raw)

            # Return a message indicating the email was saved
            return f"Email saved for {candidate.job_title} as {filename}"

        # Create tasks for all candidates
        for candidate in self.state.hydrated_candidates:
            task = asyncio.create_task(write_email(candidate))
            tasks.append(task)

        # Run all email-writing tasks concurrently and collect results
        email_results = await asyncio.gather(*tasks)

        # After all emails have been generated and saved
        print("\nAll emails have been written and saved to 'email_responses' folder.")
        for message in email_results:
            print(message)

    # =========================
    # CSV Export Helpers (NEW)
    # =========================

    def _export_top3_persona_matches(self):
        """Export top 3 persona matches per candidate in wide format."""
        if not self.state.persona_multi_matches:
            print("No persona multi-match data to export.")
            return
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
            for row in self.state.persona_multi_matches:
                m = row.matches
                cells = [row.candidate_id, row.candidate_title]
                for i in range(3):
                    if i < len(m):
                        cells.extend([m[i].persona_id, m[i].persona_name, m[i].score])
                    else:
                        cells.extend(["", "", ""])
                w.writerow(cells)
        print(f"Wrote {out.name}")

    # =============================
    # NEW: Lightweight persona matches
    # =============================
    @listen("generate_persona_matches")
    async def generate_persona_matches(self):
        from lead_score_flow.lead_types import CandidatePersonaMatches, PersonaMatchItem
        import csv

        print(
            f"Enter generate_persona_matches -> ran={self.state.persona_match_ran}, in_progress={self.state.persona_match_in_progress}"
        )
        if self.state.persona_match_ran or self.state.persona_match_in_progress:
            print("Persona matching already executed or in progress; skipping.")
            return None

        # Reentrancy guard (set both early to avoid router loops)
        self.state.persona_match_in_progress = True
        self.state.persona_match_ran = True

        if not self.state.personas:
            print("No personas loaded; cannot match.")
            # revert ran flag as nothing executed
            self.state.persona_match_ran = False
            self.state.persona_match_in_progress = False
            return None
        if not self.state.hydrated_candidates:
            print("No scored candidates yet; run scoring first.")
            # revert ran flag as nothing executed
            self.state.persona_match_ran = False
            self.state.persona_match_in_progress = False
            return None

        def simple_score(title: str, persona) -> float:
            t = set(w.lower() for w in title.split() if len(w) > 2)
            pname = persona.name.lower()
            roles = " ".join(persona.roles or []).lower()
            bag = set([w for w in (pname + " " + roles).split() if len(w) > 2])
            inter = t & bag
            if not t or not bag:
                return 0.0
            return round(100.0 * len(inter) / len(t), 1)

        top_k = 3
        all_matches: list[CandidatePersonaMatches] = []
        for sc in self.state.hydrated_candidates:
            scores = []
            title = sc.candidate.job_title
            for p in self.state.personas:
                s = simple_score(title, p)
                if s > 0:
                    scores.append((s, p))
            scores.sort(reverse=True, key=lambda x: x[0])
            top = scores[:top_k]
            items = [PersonaMatchItem(persona_id=p.id, persona_name=p.name, score=s, rationale="token_overlap") for s, p in top]
            all_matches.append(CandidatePersonaMatches(candidate_id=sc.candidate.id, candidate_title=title, matches=items))

        self.state.persona_multi_matches = all_matches

        # Narrow CSV (long form) still available if needed
        out_long = Path(__file__).parent / "persona_matches.csv"
        with out_long.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["candidate_id", "candidate_title", "persona_id", "persona_name", "score", "rationale"])
            for row in all_matches:
                if not row.matches:
                    w.writerow([row.candidate_id, row.candidate_title, "", "", 0, "no_match"])
                else:
                    for m in row.matches:
                        w.writerow([row.candidate_id, row.candidate_title, m.persona_id, m.persona_name, m.score, m.rationale])
        print(f"Lightweight persona matches exported to {out_long}")

        # Wide CSV export (top 3 columns)
        self._export_top3_persona_matches()
        # Mark complete
        self.state.persona_match_in_progress = False

        # Chain into deep dive evaluation immediately to avoid router loops
        if any(r.matches for r in all_matches):
            print("Proceeding to deep dive persona evaluation...")
            await self.deep_dive_personas()
            return None
        print("Flow complete (no matches to deep dive).")
        return None

    # =============================
    # Deep Dive Persona Evaluation
    # =============================
    @listen("persona_deepdive")
    async def deep_dive_personas(self):
        from lead_score_flow.lead_types import CandidatePersonaDeepDive
        import csv

        if self.state.deep_dive_ran or self.state.deep_dive_in_progress:
            print("Deep dive already executed or in progress; skipping.")
            return None

        self.state.deep_dive_in_progress = True

        if not self.state.persona_multi_matches:
            print("No persona multi-match results to deep dive.")
            self.state.deep_dive_in_progress = False
            return None
        results: list[CandidatePersonaDeepDive] = []

        # Map candidate id -> ScoredCandidate for context
        sc_map = {sc.candidate.id: sc for sc in self.state.hydrated_candidates}

        async def run_deepdive_for(candidate_id: str, cpm):
            sc = sc_map.get(candidate_id)
            if not sc:
                return None
            # Prepare structured input
            personas_payload = [
                {
                    "persona_id": m.persona_id,
                    "persona_name": m.persona_name,
                    "lexical_score": m.score,
                    "rationale": m.rationale,
                }
                for m in cpm.matches
            ]
            # Offline synthetic deep dive (no-LLM path)
            if not USE_LLM_DEEPDIVE:
                from lead_score_flow.lead_types import PersonaDeepDiveAssessment, CandidatePersonaDeepDive
                assessments = []
                for i, m in enumerate(cpm.matches[:3]):
                    rel = float(m.score)
                    strengths = [f"Title overlap with persona '{m.persona_name}'"] if rel > 0 else []
                    risks = ["Limited lexical overlap"] if rel < 10 else []
                    hooks = [f"Reference {sc.candidate.job_title} challenges in {sc.candidate.primary_line_of_business}"]
                    assessments.append(
                        PersonaDeepDiveAssessment(
                            persona_id=m.persona_id,
                            persona_name=m.persona_name,
                            relevance=rel,
                            strengths=strengths,
                            risks=risks,
                            messaging_hooks=hooks,
                            overall_fit="Primary contact" if i == 0 else "Secondary contact",
                            recommended=(i == 0),
                        )
                    )
                best_id = assessments[0].persona_id if assessments else ""
                return CandidatePersonaDeepDive(
                    candidate_id=candidate_id,
                    candidate_title=sc.candidate.job_title,
                    assessments=assessments,
                    best_persona_id=best_id,
                    reasoning="Selected top lexical match as best fit (offline mode).",
                )
            try:
                import asyncio as __asyncio
                crew = PersonaPipelineCrew().crew()
                result = await __asyncio.wait_for(
                    crew.kickoff_async(
                        inputs={
                            "candidate_id": candidate_id,
                            "candidate_title": sc.candidate.job_title,
                            "company_name": sc.candidate.company_name,
                            "job_function": sc.candidate.job_function,
                            "job_level": sc.candidate.job_level,
                            "line_of_business": sc.candidate.primary_line_of_business,
                            "persona_matches": personas_payload,
                        }
                    ),
                    timeout=float(os.getenv("DEEPDIVE_TIMEOUT_SECONDS", "120")),
                )
                if hasattr(result, "pydantic") and isinstance(result.pydantic, CandidatePersonaDeepDive):
                    return result.pydantic
                print(f"Deep dive returned unexpected result type for {candidate_id}")
            except Exception as e:
                print(f"Deep dive failed for candidate {candidate_id}: {e}")
            return None

        import asyncio as _asyncio
        tasks = []
        for cpm in self.state.persona_multi_matches:
            if not cpm.matches:
                continue
            tasks.append(_asyncio.create_task(run_deepdive_for(cpm.candidate_id, cpm)))
        gathered = await _asyncio.gather(*tasks)
        results = [r for r in gathered if r]
        self.state.persona_deepdive_results = results

        if not results:
            print("No deep dive results produced.")
            # Export a minimal error CSV for visibility
            out_err = Path(__file__).parent / "persona_deepdive_errors.csv"
            with out_err.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["candidate_id", "candidate_title", "error"])
                for cpm in self.state.persona_multi_matches:
                    if cpm.matches:
                        title = sc_map.get(cpm.candidate_id).candidate.job_title if sc_map.get(cpm.candidate_id) else ""
                        w.writerow([cpm.candidate_id, title, "deep_dive_failed_or_timed_out"])
            print(f"Wrote {out_err.name}")
            self.state.deep_dive_in_progress = False
            return None

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
        print(f"Wrote {out_long.name}")

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
        print(f"Wrote {out_best.name}")
        print("Deep dive complete.")
        self.state.deep_dive_ran = True
        self.state.deep_dive_in_progress = False
        return None



def kickoff():
    """Run the flow."""
    lead_score_flow = LeadScoreFlow()
    lead_score_flow.kickoff()


def plot():
    """Plot the flow."""
    lead_score_flow = LeadScoreFlow()
    lead_score_flow.plot()


if __name__ == "__main__":
    kickoff()
