import pytest
from agents.validation_agent import ValidationAgent
from agents.state import ClaimAdjudicationState

def test_validation_agent_pass():
    agent = ValidationAgent()
    state: ClaimAdjudicationState = {
        "case_data": {},
        "case_id": "TEST-VAL-01",
        "extracted_facts": {},
        "investigation_plan": [],
        "decision_dimensions": [],
        "missing_fields": [],
        "retrieved_evidence": [
            {
                "chunk_id": "CH-P07-01",
                "content": "Normal Room expenses: 1.0% of Basic Sum Insured per day.",
                "page": 7,
                "section": "SCOPE OF COVER & LIMITS",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
            }
        ],
        "retrieval_metadata": {"dense_hits": 1, "bm25_hits": 1, "rrf_candidates": 1, "reranked_top_k": 1, "top_score": 0.90},
        "coverage_assessment": {},
        "financial_breakdown": {},
        "draft_decision": "ADMISSIBLE_WITH_LIMITS",
        "confidence": 0.89,
        "key_findings": ["Room rent capped"],
        "applicable_limits": ["1% SI daily limit"],
        "missing_evidence": [],
        "citations": [
            {
                "claim": "Room rent expenses subject to 1% limit",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 7,
                "section": "SCOPE OF COVER & LIMITS",
                "chunk_id": "CH-P07-01"
            }
        ],
        "validation": {"status": "PASS", "unsupported_claims": []},
        "retry_count": 0,
        "trace": []
    }
    
    res_state = agent.process(state)
    assert res_state["validation"]["status"] == "PASS"
    assert len(res_state["validation"]["unsupported_claims"]) == 0

def test_validation_agent_fail_unsupported_claim():
    agent = ValidationAgent()
    state: ClaimAdjudicationState = {
        "case_data": {},
        "case_id": "TEST-VAL-02",
        "extracted_facts": {},
        "investigation_plan": [],
        "decision_dimensions": [],
        "missing_fields": [],
        "retrieved_evidence": [
            {
                "chunk_id": "CH-P07-01",
                "content": "Normal Room expenses: 1.0% of Basic Sum Insured.",
                "page": 7,
                "section": "SCOPE OF COVER & LIMITS",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
            }
        ],
        "retrieval_metadata": {"dense_hits": 1, "bm25_hits": 1, "rrf_candidates": 1, "reranked_top_k": 1, "top_score": 0.90},
        "coverage_assessment": {},
        "financial_breakdown": {},
        "draft_decision": "ADMISSIBLE",
        "confidence": 0.89,
        "key_findings": [],
        "applicable_limits": [],
        "missing_evidence": [],
        "citations": [
            {
                "claim": "Unrelated invalid claims statement not in policy",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 0,
                "section": "NONE",
                "chunk_id": ""
            }
        ],
        "validation": {"status": "PASS", "unsupported_claims": []},
        "retry_count": 0,
        "trace": []
    }
    
    res_state = agent.process(state)
    assert res_state["validation"]["status"] == "FAIL"
    assert res_state["draft_decision"] == "NEEDS_REVIEW"
