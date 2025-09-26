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
        # NEW: Handle OpenRouter specifics
        if key_name == "OPENROUTER_API_KEY":
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            # Optional but recommended by OpenRouter docs
            referrer = os.getenv("OPENROUTER_REFERRER")
            if referrer:
                os.environ["HTTP_REFERER"] = referrer
        break
if (USE_LLM or USE_LLM_DEEPDIVE) and not _api_key:
    print("ERROR: No API key found. Set an API key in your environment or .env file.")
    print("Tip: create a .env file alongside main.py with a line: OPENAI_API_KEY=sk-...  (restart your run after saving)")
    print("Alternatively in PowerShell:  $env:OPENAI_API_KEY=\"sk-...\"  and re-run.")
    raise SystemExit(1)
if os.getenv("LITELLM_DEBUG", "0") == "1" and _api_key:
    masked = _api_key[:3] + "..." + _api_key[-4:]
    src = _api_key_source or "(unknown env var)"
    print(f"LLM enabled. Using {_api_key_source}: {masked}")
    # Print selected model for routing visibility
    _llm_model = os.getenv("LLM_MODEL")
    if _llm_model:
        print(f"LLM_MODEL={_llm_model}")
    # Add this line to debug the key
    print(f"DEBUG: Using key from {_api_key_source} starting with '{_api_key[:7]}' and ending with '{_api_key[-4:]}'")
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
    elif _api_key_source == "OPENROUTER_API_KEY":
        print("OpenRouter API key detected. LiteLLM will use the OpenRouter endpoint.")
        print("Set LLM_MODEL to the desired model, e.g., 'openrouter/google/gemini-flash-1.5'")
    else:
        print("Note: Using a non-standard key env var; ensure your provider config is correct.")
    print("Set API_SANITY_CHECK=0 to silence this.")

import asyncio
from typing import List
from pathlib import Path

from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field
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


from uuid import uuid4, UUID

class LeadScoreState(BaseModel):
    id: UUID = Field(default_factory=uuid4)
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
    # Router dispatch guard to prevent repeated scheduling of the same event
    persona_match_event_dispatched: bool = False

    # NEW: best persona per lead (computed after scoring)
    # Removed: single best persona matches no longer needed
    lead_persona_matches: List[LeadPersonaMatch] = []  # kept for backward compatibility (unused)


class LeadScoreFlow(Flow[LeadScoreState]):
    initial_state = LeadScoreState

    @start()
    def load_leads(self):
        print("\n[FLOW_STEP] ==> @start: load_leads")
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
            # search project root for persona JSON exports
            project_root = Path(__file__).parent.parent.parent
            root_json = list(project_root.glob("*.json"))
            # Filter likely persona files (contain 'INOV' or have 'Provider'/'Payer' etc.)
            persona_like = [f for f in root_json if any(k in f.name for k in ["INOV", "Provider", "Payer", "Pharmacy", "Insights"]) ] or root_json
            if persona_like:
                loaded_auto = _load_many(persona_like)
                if loaded_auto:
                    self.state.personas = loaded_auto
                    self.state.persona_folder = str(project_root)
                    print(f"Auto-loaded {len(loaded_auto)} persona(s) from project root: {project_root}")

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
        print("[FLOW_STEP] <== Exiting: load_leads")

    @listen(or_(load_leads, "scored_leads_feedback"))
    async def score_leads(self):
        print("\n[FLOW_STEP] ==> @listen: score_leads")
        print("Scoring leads")
        tasks = []
        import json

        async def score_single_candidate(candidate: Candidate):
            from lead_score_flow.lead_types import CandidateScore
            try:
                if not USE_LLM:
                    # Offline deterministic score (cheap baseline)
                    jt = (candidate.job_title or "").lower()
                    base = 10.0
                    if "director" in jt or "vp" in jt or "chief" in jt:
                        base = 35.0
                    elif "manager" in jt:
                        base = 25.0
                    return CandidateScore(candidate=candidate, score=base)

                result = await (
                    LeadScoreCrew()
                    .crew()
                    .kickoff_async(
                        inputs={
                            "candidate_id": candidate.id,
                            "name": candidate.job_title,
                            "bio": candidate.company_name,
                            "job_description": JOB_DESCRIPTION,
                            "additional_instructions": self.state.scored_leads_feedback,
                        }
                    )
                )
                
                # First, try to use the pydantic object directly
                if result.pydantic and isinstance(result.pydantic, CandidateScore):
                    return result.pydantic

                # Fallback: If pydantic fails, parse the raw string output
                if isinstance(result.raw, str):
                    print("Pydantic conversion failed, attempting to parse raw output.")
                    data = json.loads(result.raw)
                    # The crew might return the candidate as a nested JSON string, so parse again if needed
                    if 'candidate' in data and isinstance(data['candidate'], str):
                        data['candidate'] = json.loads(data['candidate'])
                    
                    # FIX: Manually cast candidate ID to string to prevent validation error
                    if 'candidate' in data and isinstance(data['candidate'], dict):
                        # Sanitize all None values in the nested candidate dictionary
                        for key, value in data['candidate'].items():
                            if value is None:
                                data['candidate'][key] = ""
                        if 'id' in data['candidate']:
                            data['candidate']['id'] = str(data['candidate']['id'])
                        
                    return CandidateScore(**data)
                
                raise ValueError("Crew returned no usable output (pydantic or raw).")

            except Exception as e:
                # Log error and return a failed score for this candidate
                print(f"Error scoring candidate {candidate.id} ({candidate.job_title}): {e}")
                return CandidateScore(candidate=candidate, score=0.0)

        for candidate in self.state.candidates:
            print("Scoring candidate:", candidate.job_title)
            task = asyncio.create_task(score_single_candidate(candidate))
            tasks.append(task)

        candidate_scores = await asyncio.gather(*tasks)
        # Filter out any None results from catastrophic failures before appending
        self.state.candidate_score = [score for score in candidate_scores if score is not None]
        print("Finished scoring leads: ", len(self.state.candidate_score))
        print("[FLOW_STEP] <== Exiting: score_leads")

    @router(score_leads)
    async def human_in_the_loop(self):
        print("\n[FLOW_STEP] ==> @router: human_in_the_loop (NOW ASYNC)")
        """
        This router is now async. It calls the persona pipeline directly 
        instead of dispatching an event, which solves the infinite loop.
        """
        print("Hydrating candidates and exporting scores...")

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

        # --- Sequential Logic ---
        # If personas exist and the pipeline hasn't been run, run it directly.
        if self.state.personas and not self.state.persona_match_ran:
            print("Router: Directly calling the persona pipeline now.")
            self.state.persona_match_ran = True  # Prevent re-runs
            # Pass the original, complete candidates list to the pipeline
            await self.run_persona_pipeline(candidates_for_pipeline=self.state.candidates)
        elif not self.state.personas:
            print("Router: No personas loaded. Skipping persona analysis.")
        else:
            print("Router: Persona pipeline has already been completed.")

        # After the pipeline (or skipping it), the flow is complete.
        print(f"[FLOW_STEP] <== Exiting: human_in_the_loop, returning 'flow_complete'")
        return "flow_complete"

    # DECORATOR REMOVED - This is now a regular method, not a listener
    async def run_persona_pipeline(self, candidates_for_pipeline: List[Candidate]):
        """
        This is now a regular method called directly by the router.
        It combines the logic of the old `generate_persona_matches`
        and `deep_dive_personas` into a single, sequential pipeline.
        It receives a clean list of candidates to ensure data integrity.
        """
        print("\n[FLOW_STEP] ==> Now a regular method: run_persona_pipeline (Unified)")

        # Idempotency Guard: Prevent re-running the whole pipeline
        if self.state.persona_match_in_progress:
            print("Persona pipeline is already in progress. Skipping.")
            return

        # Set flags to indicate the pipeline is running
        self.state.persona_match_in_progress = True

        # =================================================
        # 1. Lightweight Persona Matching
        # =================================================
        print("\n[PIPELINE_SUB-STEP] Starting: Lightweight Persona Matching")
        from lead_score_flow.lead_types import CandidatePersonaMatches, PersonaMatchItem
        import csv

        def simple_score(title: str, persona) -> float:
            t = set(w.lower() for w in title.split() if len(w) > 2)
            pname = persona.name.lower()
            roles = " ".join(persona.roles or []).lower()
            bag = set([w for w in (pname + " " + roles).split() if len(w) > 2])
            inter = t & bag
            if not t or not bag: return 0.0
            return round(100.0 * len(inter) / len(t), 1)

        all_matches: list[CandidatePersonaMatches] = []
        # Use the clean candidate list passed into this method
        for cand in candidates_for_pipeline:
            scores = sorted(
                [(simple_score(cand.job_title, p), p) for p in self.state.personas if simple_score(cand.job_title, p) > 0],
                reverse=True, key=lambda x: x[0]
            )
            top_items = [PersonaMatchItem(persona_id=p.id, persona_name=p.name, score=s, rationale="token_overlap") for s, p in scores[:3]]
            all_matches.append(CandidatePersonaMatches(candidate_id=cand.id, candidate_title=cand.job_title, matches=top_items))

        self.state.persona_multi_matches = all_matches
        self._export_top3_persona_matches() # Export wide CSV
        print("Lightweight persona matching complete. Results exported.")


        # =================================================
        # 2. Deep Dive Analysis
        # =================================================
        if not USE_LLM_DEEPDIVE:
            print("\n[PIPELINE_SUB-STEP] Skipping: Deep Dive (USE_LLM_DEEPDIVE is false)")
        else:
            print("\n[PIPELINE_SUB-STEP] Starting: Deep Dive Analysis")
            # Pass the clean candidate list to the deep dive logic as well
            await self._execute_deep_dive_logic(candidates_for_pipeline=candidates_for_pipeline)


        # Mark the entire pipeline as complete
        self.state.persona_match_in_progress = False
        print("\n[FLOW_STEP] <== Exiting: run_persona_pipeline. Control returns to human_in_the_loop.")
        # No return value needed as it's not a listener anymore
        

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

    async def _execute_deep_dive_logic(self, candidates_for_pipeline: List[Candidate]):
        """Helper containing the implementation from the old `deep_dive_personas`."""
        from lead_score_flow.lead_types import CandidatePersonaDeepDive, PersonaDeepDiveAssessment
        import csv
        import json
        import asyncio

        if self.state.deep_dive_ran or self.state.deep_dive_in_progress:
            return
        self.state.deep_dive_in_progress = True

        # Use the clean candidate list passed into this method
        sc_map = {c.id: c for c in candidates_for_pipeline}
        tasks = []

        async def run_deepdive_for(cpm: CandidatePersonaMatches):
            sc = sc_map.get(cpm.candidate_id)
            if not sc: return None

            personas_payload = [{"persona_id": m.persona_id, "persona_name": m.persona_name, "lexical_score": m.score} for m in cpm.matches]
            personas_payload_json = json.dumps(personas_payload, ensure_ascii=False)

            try:
                crew = PersonaPipelineCrew().crew()
                # Sanitize inputs to prevent errors with None values
                inputs = {
                    "candidate_id": cpm.candidate_id,
                    "candidate_title": sc.job_title or "",
                    "company_name": sc.company_name or "",
                    "job_function": sc.job_function or "",
                    "job_level": sc.job_level or "",
                    "line_of_business": sc.primary_line_of_business or "",
                    "persona_matches_json": personas_payload_json,
                }
                result = await asyncio.wait_for(
                    crew.kickoff_async(inputs=inputs),
                    timeout=float(os.getenv("DEEPDIVE_TIMEOUT_SECONDS", "120")),
                )
                if hasattr(result, "pydantic") and isinstance(result.pydantic, CandidatePersonaDeepDive):
                    return result.pydantic
            except Exception as e:
                print(f"Deep dive failed for candidate {cpm.candidate_id}: {e}")
            return None

        for cpm in self.state.persona_multi_matches:
            if cpm.matches:
                tasks.append(asyncio.create_task(run_deepdive_for(cpm)))

        gathered = await asyncio.gather(*tasks)
        results = [r for r in gathered if r]
        self.state.persona_deepdive_results = results

        if results:
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
            print(f"Deep dive analysis complete. Found {len(results)} results.")
        else:
            print("No deep dive results were produced.")

        self.state.deep_dive_ran = True
        self.state.deep_dive_in_progress = False

    def flow_complete(self):
        print("\n[FLOW_STEP] ==> flow_complete (TERMINAL)")
        print("[FLOW_STEP] <== Exiting: flow_complete. The flow has ended.")
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
