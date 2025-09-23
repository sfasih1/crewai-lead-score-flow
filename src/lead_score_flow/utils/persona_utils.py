# src/lead_score_flow/utils/persona_utils.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# Types your project already has:
# - Persona: includes .industries (we mirror segment into this), .roles, etc.
# - ScoredCandidate: includes .score (0..100) and typical attributes like name, role/title, industry/segment, etc.
from lead_score_flow.lead_types import Persona, ScoredCandidate

# --------- facet helpers ---------

def _norm(v: Optional[str]) -> str:
    return (v or "").strip().lower()

def _pct(x: float) -> float:
    return float(max(0.0, min(100.0, x)))

def _score_industry(lead: ScoredCandidate, persona: Persona) -> float:
    """Exact industry/segment match -> 100; substring/cousin -> 70; else 0; neutral 50 if unknown."""
    pins = {i.lower() for i in (persona.industries or [])}
    if not pins:
        return 50.0
    lind = _norm(getattr(lead, "industry", None) or getattr(lead, "segment", None))
    if not lind:
        return 50.0
    if lind in pins:
        return 100.0
    for p in pins:
        if p in lind or lind in p:
            return 70.0
    return 0.0

def _score_role(lead: ScoredCandidate, persona: Persona) -> float:
    """Exact role/title match -> 100; partial ladder/substring -> 80; else 0; neutral 50 if unknown."""
    pros = {r.lower() for r in (persona.roles or [])}
    if not pros:
        return 50.0
    lrole = _norm(getattr(lead, "role", None) or getattr(lead, "title", None) or getattr(lead, "bio", None))
    if not lrole:
        return 50.0
    if lrole in pros:
        return 100.0
    for r in pros:
        if r in lrole or lrole in r:
            return 80.0
    return 0.0

def _score_size(lead: ScoredCandidate, persona: Persona) -> float:
    """Company size fit; neutral if unknown on either side."""
    rng = getattr(persona, "employee_count_range", None)
    n = getattr(lead, "employee_count", None) or getattr(lead, "company_size", None)
    if not rng or (rng[0] is None and rng[1] is None) or not n:
        return 50.0
    lo = rng[0] if rng[0] is not None else n
    hi = rng[1] if rng[1] is not None else n
    if lo <= n <= hi:
        return 100.0
    if n < lo:
        # linear decay below range
        return _pct(100.0 * max(0.0, 1.0 - (lo - n) / max(1.0, lo)))
    # n > hi: linear decay above range
    return _pct(100.0 * max(0.0, 1.0 - (n - hi) / max(1.0, hi)))

def _score_region(lead: ScoredCandidate, persona: Persona) -> float:
    """Region match; neutral if unknown."""
    pros = {r.lower() for r in (getattr(persona, "regions", None) or [])}
    if not pros:
        return 50.0
    lreg = _norm(getattr(lead, "region", None) or getattr(lead, "location", None))
    if not lreg:
        return 50.0
    if lreg in pros:
        return 100.0
    for r in pros:
        if r in lreg or lreg in r:
            return 70.0
    return 0.0

def _score_tech(lead: ScoredCandidate, persona: Persona) -> float:
    """Jaccard overlap of tech keywords; neutral if unknown."""
    pks = {t.lower() for t in (getattr(persona, "tech_stack_keywords", None) or [])}
    lks = getattr(lead, "tech_stack", None) or getattr(lead, "keywords", None)
    if not pks:
        return 50.0
    if not lks:
        return 50.0
    if isinstance(lks, str):
        lset = {s.strip().lower() for s in lks.split("|") if s.strip()}
    else:
        lset = {str(s).strip().lower() for s in lks if str(s).strip()}
    inter = len(pks & lset)
    union = len(pks | lset) or 1
    return _pct(100.0 * inter / union)

def _score_budget(lead: ScoredCandidate, persona: Persona) -> float:
    """Budget check; neutral if unknown."""
    pmin = getattr(persona, "budget_min", None)
    lbud = getattr(lead, "estimated_budget", None)
    if not pmin or not lbud:
        return 50.0
    if lbud >= pmin:
        return 100.0
    return _pct(80.0 * (float(lbud) / float(pmin)))

def _score_compliance(lead: ScoredCandidate, persona: Persona) -> float:
    """Compliance needs; neutral if unknown; penalize missing must-haves."""
    needs = {c.lower() for c in (getattr(persona, "compliance_needs", None) or [])}
    lcomp = getattr(lead, "compliance", None)
    if not needs:
        return 50.0
    if not lcomp:
        return 50.0
    if isinstance(lcomp, str):
        lset = {s.strip().lower() for s in lcomp.split("|") if s.strip()}
    else:
        lset = {str(s).strip().lower() for s in lcomp if str(s).strip()}
    if needs.issubset(lset):
        return 100.0
    if needs & lset:
        return 70.0
    return 40.0

# --------- similarity & alignment ---------

# Default emphasis for the facets you actually have (industry/segment + roles).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "industry": 0.50,
    "role": 0.50,
    "size": 0.00,
    "region": 0.00,
    "tech": 0.00,
    "budget": 0.00,
    "compliance": 0.00,
}

def compute_similarity(
    lead: ScoredCandidate,
    persona: Persona,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Return (similarity 0..100, component_scores) using dynamic reweighting:
    only facets that are INFORMATIVE for this (lead, persona) pair are used,
    then weights are renormalized to sum to 1.
    """
    parts = {
        "industry": _score_industry(lead, persona),
        "role": _score_role(lead, persona),
        "size": _score_size(lead, persona),
        "region": _score_region(lead, persona),
        "tech": _score_tech(lead, persona),
        "budget": _score_budget(lead, persona),
        "compliance": _score_compliance(lead, persona),
    }

    # A facet is informative only if BOTH sides provide data
    informative = {
        "industry": bool(persona.industries) and bool(getattr(lead, "industry", None) or getattr(lead, "segment", None)),
        "role": bool(persona.roles) and bool(getattr(lead, "role", None) or getattr(lead, "title", None) or getattr(lead, "bio", None)),
        "size": getattr(persona, "employee_count_range", None) is not None and bool(getattr(lead, "employee_count", None) or getattr(lead, "company_size", None)),
        "region": bool(getattr(persona, "regions", None)) and bool(getattr(lead, "region", None) or getattr(lead, "location", None)),
        "tech": bool(getattr(persona, "tech_stack_keywords", None)) and bool(getattr(lead, "tech_stack", None) or getattr(lead, "keywords", None)),
        "budget": getattr(persona, "budget_min", None) is not None and bool(getattr(lead, "estimated_budget", None)),
        "compliance": bool(getattr(persona, "compliance_needs", None)) and bool(getattr(lead, "compliance", None)),
    }

    w = (getattr(persona, "priority_weights", None) or weights or DEFAULT_WEIGHTS).copy()
    active = {k: v for k, v in w.items() if informative.get(k, False) and v > 0}
    if not active:
        active = {"industry": 0.5, "role": 0.5}  # safe fallback

    denom = sum(active.values()) or 1.0
    norm = {k: v / denom for k, v in active.items()}

    sim = sum(norm[k] * parts[k] for k in norm)
    return _pct(sim), parts

def compute_alignment_score(
    similarity: float,
    lead_score: float,
    persona_goodness: Optional[float] = None,
    *,
    alpha: float = 0.60,  # similarity weight
    beta: float  = 0.30,  # lead score weight
    gamma: float = 0.10   # persona "goodness" if you later compute it
) -> float:
    pg = 50.0 if persona_goodness is None else float(persona_goodness)
    val = (alpha * float(similarity)) + (beta * float(lead_score)) + (gamma * pg)
    return _pct(val)

def best_persona_for_lead(
    lead: ScoredCandidate,
    personas: List[Persona],
    persona_goodness_map: Optional[Dict[str, float]] = None
) -> Tuple[str, str, float, float]:
    """
    Returns (persona_id, persona_name, similarity, alignment) for the best persona.
    Falls back to zeros if no personas are provided.
    """
    best: Optional[Tuple[str, str, float, float]] = None
    for p in personas:
        sim, _ = compute_similarity(lead, p)
        pg = persona_goodness_map.get(p.id) if persona_goodness_map else None
        align = compute_alignment_score(similarity=sim, lead_score=float(lead.score or 0.0), persona_goodness=pg)
        if (best is None) or (align > best[-1]):
            best = (p.id, p.name, float(sim), float(align))
    return best if best is not None else ("", "", 0.0, 0.0)
