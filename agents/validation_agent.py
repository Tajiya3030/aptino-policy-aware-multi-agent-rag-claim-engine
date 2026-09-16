import time
from typing import Dict, Any, List
from agents.state import ClaimAdjudicationState, ExecutionTrace, ValidationResult

class ValidationAgent:
    """Agent 5: Independent auditor. Verifies citation grounding against policy chunks and enforces PASS/FAIL."""

    def process(self, state: ClaimAdjudicationState) -> ClaimAdjudicationState:
        t0 = time.time()
        citations = state.get("citations", [])
        evidence = state.get("retrieved_evidence", [])
        decision = state.get("draft_decision", "")

        unsupported = []
        evidence_text_blob = " ".join([e.get("content", "").lower() for e in evidence])

        for cite in citations:
            claim = cite.get("claim", "")
            chunk_id = cite.get("chunk_id", "")
            page = cite.get("page", 0)

            # Check if citation page & chunk_id are valid
            if page <= 0 or not chunk_id:
                unsupported.append(f"Citation missing valid page or chunk_id provenance: {claim}")
                continue

            # Verify key words from claim appear in retrieved policy evidence
            claim_words = [w.lower() for w in claim.split() if len(w) > 4]
            matches = [w for w in claim_words if w in evidence_text_blob]
            
            if len(claim_words) > 0 and len(matches) / len(claim_words) < 0.20:
                unsupported.append(f"Claim text unsupported by policy evidence: '{claim}'")

        if unsupported and decision not in ["NEEDS_REVIEW", "NOT_ADMISSIBLE"]:
            validation_res: ValidationResult = {
                "status": "FAIL",
                "unsupported_claims": unsupported
            }
            # Switch to NEEDS_REVIEW due to citation grounding audit failure
            state["draft_decision"] = "NEEDS_REVIEW"
            state["missing_evidence"].append("Citations failed grounding audit. Switched to NEEDS_REVIEW.")
        else:
            validation_res: ValidationResult = {
                "status": "PASS",
                "unsupported_claims": []
            }

        state["validation"] = validation_res

        latency = int((time.time() - t0) * 1000)
        trace_entry: ExecutionTrace = {
            "agent": "ValidationAgent",
            "action": f"Validation Audit outcome: {validation_res['status']}",
            "retrieved_chunks": 0,
            "reranked_chunks": 0,
            "latency_ms": latency
        }
        state["trace"].append(trace_entry)
        return state
