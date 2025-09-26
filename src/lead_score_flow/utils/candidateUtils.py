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

    # Create a dictionary of the original candidates for easy lookup
    candidate_dict = {c.id: c for c in candidates}

    scored_candidates: List[ScoredCandidate] = []
    for score_id, score_obj in score_dict.items():
        # Use the original candidate data from candidate_dict
        original_candidate = candidate_dict.get(score_id)
        if original_candidate:
            reason = getattr(score_obj, 'reason', '')
            # Create the ScoredCandidate with the complete, original candidate object
            scored_candidates.append(ScoredCandidate(candidate=original_candidate, score=score_obj.score, reason=reason))
        else:
            # This case should ideally not happen if the lists are in sync
            scored_candidates.append(ScoredCandidate(candidate=score_obj.candidate, score=score_obj.score, reason="Original candidate not found"))

    # Add any candidates that were not scored
    for cand in candidates:
        if cand.id not in score_dict:
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
