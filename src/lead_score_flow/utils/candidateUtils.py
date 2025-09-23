from typing import List

from lead_score_flow.lead_types import Candidate, CandidateScore, ScoredCandidate
import csv


def combine_candidates_with_scores(
    candidates: List[Candidate], candidate_scores: List[CandidateScore]
) -> List[ScoredCandidate]:
    """
    Combine the candidates with their scores using a dictionary for efficient lookups.
    """
    print("COMBINING CANDIDATES WITH SCORES")
    print("SCORES:", candidate_scores)
    print("CANDIDATES:", candidates)
    # Map candidate.id -> CandidateScore
    score_dict = {cs.candidate.id: cs for cs in candidate_scores if cs and cs.candidate}
    print("SCORE DICT KEYS:", list(score_dict.keys()))

    scored_candidates: List[ScoredCandidate] = []
    for cand in candidates:
        cs = score_dict.get(cand.id)
        if cs:
            # ScoredCandidate model expects (candidate, score, reason)
            reason = getattr(cs, 'reason', '')
            scored_candidates.append(ScoredCandidate(candidate=cand, score=cs.score, reason=reason))
        else:
            scored_candidates.append(ScoredCandidate(candidate=cand, score=0.0, reason="Not scored (API error or rate limit)"))

    print("SCORED CANDIDATES (count):", len(scored_candidates))

    # Export basic scores (persona-enhanced export now handled elsewhere in main flow)
    with open("lead_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "job_title", "company_name", "score", "reason"])
        for sc in scored_candidates:
            writer.writerow([
                sc.candidate.id,
                sc.candidate.job_title,
                sc.candidate.company_name,
                sc.score,
                sc.reason,
            ])
    print("(candidateUtils) Basic lead scores saved to lead_scores.csv")
    return scored_candidates
