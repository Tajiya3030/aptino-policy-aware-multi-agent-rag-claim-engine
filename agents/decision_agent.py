import time
from typing import Dict, Any, List
from agents.state import ClaimAdjudicationState, ExecutionTrace, Citation

class DecisionAgent:
    """Agent 4: Synthesizes adjudication decision status, citations, findings, and confidence score."""

    def process(self, state: ClaimAdjudicationState) -> ClaimAdjudicationState:
        t0 = time.time()
        facts = state["extracted_facts"]
        assessment = state["coverage_assessment"]
        missing_fields = state["missing_fields"]
        evidence = state["retrieved_evidence"]
        meta = state.get("retrieval_metadata", {})

        key_findings = []
        applicable_limits = []
        citations: List[Citation] = []
        missing_evidence = []

        # Check Abstention mechanism (NEEDS_REVIEW)
        if missing_fields or facts.get("evidence_context", {}).get("hospital_registered") is None and facts.get("treatment_type") == "inpatient" and not facts.get("network_provider"):
            decision = "NEEDS_REVIEW"
            confidence = 0.50
            if missing_fields:
                missing_evidence.extend(missing_fields)
            else:
                missing_evidence.append("Hospital registration and minimum bed criteria documentation missing")
            
            key_findings.append("A safe final decision cannot be made because required policy evidence or hospital registration criteria is unverified.")
            
            citations.append({
                "claim": "Hospital must meet Clinical Establishments Act registration or minimum 10/15 beds criteria.",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 3,
                "section": "DEFINITIONS & ELIGIBILITY",
                "chunk_id": "CH-P03-04"
            })
        elif not assessment["is_covered"]:
            decision = "NOT_ADMISSIBLE"
            confidence = 0.92
            
            reason = assessment.get("exclusion_triggered") or assessment.get("waiting_period_violation") or "Claim excluded under policy terms."
            key_findings.append(f"Claim is not admissible: {reason}")
            
            if assessment.get("waiting_period_violation"):
                citations.append({
                    "claim": f"Waiting period condition violated: {assessment['waiting_period_violation']}",
                    "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                    "page": 9,
                    "section": "EXCLUSIONS & WAITING PERIODS",
                    "chunk_id": "CH-P09-02"
                })
            elif assessment.get("exclusion_triggered"):
                citations.append({
                    "claim": f"Treatment excluded under policy: {assessment['exclusion_triggered']}",
                    "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                    "page": 10,
                    "section": "EXCLUSIONS & WAITING PERIODS",
                    "chunk_id": "CH-P10-01"
                })
        else:
            # Claim is covered. Check deductions & limits
            deductions = assessment.get("sublimit_deductions", [])
            if deductions:
                decision = "ADMISSIBLE_WITH_LIMITS"
                confidence = 0.89
                key_findings.append("Hospitalization treatment is covered subject to policy category sublimits and room rent caps.")
                
                for d in deductions:
                    applicable_limits.append(f"{d['category']} limit applied: {d['reason']} (Deduction: INR {d['deduction']:.0f})")
                
                citations.append({
                    "claim": "Room rent and hospital charges subject to daily sub-limits (1% normal / 2% ICU) and fee caps.",
                    "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                    "page": 7,
                    "section": "SCOPE OF COVER & LIMITS",
                    "chunk_id": "CH-P07-01"
                })
            else:
                decision = "ADMISSIBLE"
                confidence = 0.95
                key_findings.append("Treatment and hospitalization expenses are fully admissible within basic Sum Insured.")
                
                citations.append({
                    "claim": "Inpatient hospitalization expenses covered in full up to Basic Sum Insured.",
                    "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                    "page": 7,
                    "section": "SCOPE OF COVER & LIMITS",
                    "chunk_id": "CH-P07-01"
                })

        # Multi-signal confidence calculation
        # Signal 1: Top cross-encoder reranker score (0.40 weight)
        top_rerank_score = meta.get("top_score", 0.80)
        signal_retrieval = min(1.0, max(0.5, top_rerank_score)) * 0.40
        
        # Signal 2: Evidence count (0.30 weight)
        signal_evidence = min(1.0, len(evidence) / 5.0) * 0.30

        # Signal 3: Decision determinism & Missing evidence penalty (0.30 weight)
        if decision == "NEEDS_REVIEW":
            signal_determinism = 0.10
        elif missing_evidence:
            signal_determinism = 0.15
        else:
            signal_determinism = 0.30

        computed_confidence = round(signal_retrieval + signal_evidence + signal_determinism, 2)

        state["draft_decision"] = decision
        state["confidence"] = computed_confidence
        state["key_findings"] = key_findings
        state["applicable_limits"] = applicable_limits
        state["missing_evidence"] = missing_evidence
        state["citations"] = citations

        latency = int((time.time() - t0) * 1000)
        trace_entry: ExecutionTrace = {
            "agent": "DecisionAgent",
            "action": f"Synthesized final status '{decision}' with confidence {computed_confidence} and {len(citations)} citations",
            "retrieved_chunks": 0,
            "reranked_chunks": 0,
            "latency_ms": latency
        }
        state["trace"].append(trace_entry)
        return state
