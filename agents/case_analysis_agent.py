import time
from typing import Dict, Any
from agents.state import ClaimAdjudicationState, ExecutionTrace

class CaseAnalysisAgent:
    """Agent 1: Extracts claim facts, determines decision dimensions, and detects missing fields."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def process(self, state: ClaimAdjudicationState) -> ClaimAdjudicationState:
        t0 = time.time()
        c = state["case_data"]
        case_id = c.get("case_id", "UNKNOWN")
        
        patient = c.get("patient", {})
        hospital = c.get("hospital", {})
        treatment = c.get("treatment", {})
        expenses = c.get("expenses_inr", {})
        documents = c.get("documents", [])
        evidence_context = c.get("evidence_context", {})
        prior_policy = c.get("prior_policy", {})

        # Fact Extraction
        facts = {
            "case_id": case_id,
            "sum_insured": c.get("sum_insured_inr", 0),
            "continuous_coverage_months": c.get("continuous_coverage_months", 0),
            "prior_insurer_continuous_years": c.get("prior_insurer_continuous_years", 0),
            "diagnosis": treatment.get("diagnosis", ""),
            "procedure": treatment.get("procedure", ""),
            "treatment_type": treatment.get("type", "inpatient"),
            "admission_hours": treatment.get("admission_hours", 24),
            "pre_existing": treatment.get("pre_existing", False),
            "experimental": treatment.get("experimental", False),
            "network_provider": hospital.get("network_provider", True),
            "documents_submitted": documents,
            "evidence_context": evidence_context,
            "prior_policy": prior_policy
        }

        # Identify key decision dimensions
        dimensions = []
        if facts["continuous_coverage_months"] < 1:
            dimensions.append("initial_waiting_period_30_days")
        if facts["pre_existing"]:
            dimensions.append("pre_existing_diseases_48_months")
        if facts["experimental"]:
            dimensions.append("experimental_unproven_exclusion")
        if facts["treatment_type"] == "domiciliary":
            dimensions.append("domiciliary_hospitalization_sublimit")
        if facts["treatment_type"] == "day_care" or facts["admission_hours"] < 24:
            dimensions.append("day_care_24_hour_waiver")
        if "Cosmetic" in facts["procedure"] or "Cosmetic" in facts["diagnosis"]:
            dimensions.append("cosmetic_surgery_exclusion")
        
        # Room rent & limit dimensions
        dimensions.append("room_rent_and_category_sublimits")
        dimensions.append("pre_post_hospitalization_windows")
        dimensions.append("hospital_definition_compliance")

        # Missing Fields / Evidence Check
        missing_fields = []
        if evidence_context:
            if evidence_context.get("hospital_registered") is None or evidence_context.get("medical_necessity_confirmed") is None or evidence_context.get("hospital_minimum_criteria_documented") is False:
                missing_fields.append("hospital_registration_or_medical_necessity_verification")

        if "itemized_bill" not in documents and "discharge_summary" not in documents and "procedure_record" not in documents:
            missing_fields.append("itemized_hospital_bill_or_discharge_summary")

        state["case_id"] = case_id
        state["extracted_facts"] = facts
        state["decision_dimensions"] = dimensions
        state["missing_fields"] = missing_fields
        state["investigation_plan"] = [f"Retrieve policy terms for dimension: {d}" for d in dimensions]
        
        latency = int((time.time() - t0) * 1000)
        trace_entry: ExecutionTrace = {
            "agent": "CaseAnalysisAgent",
            "action": f"Extracted facts & mapped {len(dimensions)} decision dimensions",
            "retrieved_chunks": 0,
            "reranked_chunks": 0,
            "latency_ms": latency
        }
        state["trace"].append(trace_entry)
        return state
