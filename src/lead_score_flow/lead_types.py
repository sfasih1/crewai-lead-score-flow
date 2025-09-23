# Renamed from types.py to lead_types.py
from typing import List, Optional, Dict
from pydantic import BaseModel

class LeadPersonaMatch(BaseModel):
    lead_id: str
    lead_name: str
    best_persona_id: str
    best_persona_name: str
    similarity_score: float          # 0..100, deterministic similarity
    alignment_score: float           # 0..100, blended score (see util below)

# NEW: Lightweight multi-match structures (non-LLM deterministic matching)
class PersonaMatchItem(BaseModel):
    persona_id: str
    persona_name: str
    score: float  # 0..100 simple similarity
    rationale: str = ""  # optionally how score derived

class CandidatePersonaMatches(BaseModel):
    candidate_id: str
    candidate_title: str
    matches: List[PersonaMatchItem] = []

# Deep-dive LLM evaluation structures
class PersonaDeepDiveAssessment(BaseModel):
    persona_id: str
    persona_name: str
    relevance: float  # 0..100 subjective relevance from LLM
    strengths: List[str] = []
    risks: List[str] = []
    messaging_hooks: List[str] = []
    overall_fit: str = ""  # short phrase summary
    recommended: bool = False  # true for exactly one best persona per candidate

class CandidatePersonaDeepDive(BaseModel):
    candidate_id: str
    candidate_title: str
    assessments: List[PersonaDeepDiveAssessment]
    best_persona_id: str
    reasoning: str = ""  # justification for best persona selection

class Candidate(BaseModel):
    id: str
    job_title: str
    company_name: str
    primary_line_of_business: str
    primary_line_of_business_segment: str
    primary_line_of_business_unique_id: str
    job_title_department: str
    job_title_c: str
    job_function: str
    job_function_segment: str
    job_level: str
    job_level_segment: str
    job_segment: str
    role: str
    role_1: str
    secondary_line_of_business: str
    source_system: str
    score: float = 0.0

class CandidateScore(BaseModel):
    candidate: Candidate
    score: float

class ScoredCandidate(BaseModel):
    candidate: Candidate
    score: float
    reason: str

class Persona(BaseModel):
    id: str
    name: str
    # We’ll accept either “Business Unit”, “Segment”, or “Industry” from the doc
    segment: Optional[str] = None
    industries: List[str] = []            # we will mirror segment into this for compatibility
    roles: List[str] = []                 # parsed from Titles/Roles section
    description: Optional[str] = ""       # optional, if found
    link: Optional[str] = ""              # optional (e.g., from doc)
    # Not provided by PDFs typically, but kept for future enrichment:
    employee_count_range: Optional[List[Optional[int]]] = None
    regions: List[str] = []
    tech_stack_keywords: List[str] = []
    budget_min: Optional[float] = None
    pain_points: List[str] = []
    compliance_needs: List[str] = []
    priority_weights: Optional[Dict[str, float]] = None