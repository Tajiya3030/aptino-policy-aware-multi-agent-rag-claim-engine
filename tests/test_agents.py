import pytest
from agents.workflow import ClaimAdjudicationWorkflow

def test_pub001_admissible_with_limits():
    wf = ClaimAdjudicationWorkflow()
    case_pub1 = {
        "case_id": "PUB-001",
        "sum_insured_inr": 500000,
        "continuous_coverage_months": 14,
        "patient": {"age": 34},
        "hospital": {"network_provider": True},
        "treatment": {"type": "inpatient", "admission_hours": 96, "diagnosis": "Acute appendicitis", "procedure": "Appendectomy"},
        "expenses_inr": {"room": 30000, "doctor_fees": 30000, "medicines_diagnostics": 90000, "pre_hospitalization": 5000, "post_hospitalization": 7000, "ambulance": 1200},
        "documents": ["claim_form", "discharge_summary", "itemized_bill", "doctor_prescription"]
    }
    res = wf.run(case_pub1)
    assert res["decision"] == "ADMISSIBLE_WITH_LIMITS"
    assert len(res["applicable_limits"]) >= 1
    assert res["validation"]["status"] == "PASS"

def test_pub006_needs_review_abstention():
    wf = ClaimAdjudicationWorkflow()
    case_pub6 = {
        "case_id": "PUB-006",
        "sum_insured_inr": 500000,
        "continuous_coverage_months": 30,
        "patient": {"age": 38},
        "hospital": {"name": "Prime Hospital", "network_provider": True},
        "treatment": {"type": "inpatient", "admission_hours": 96, "diagnosis": "Acute infection"},
        "expenses_inr": {"room": 30000, "doctor_fees": 20000, "medicines_diagnostics": 100000},
        "documents": ["claim_form", "discharge_summary"],
        "evidence_context": {"hospital_registered": None, "medical_necessity_confirmed": None}
    }
    res = wf.run(case_pub6)
    assert res["decision"] == "NEEDS_REVIEW"
    assert len(res["missing_evidence"]) >= 1
