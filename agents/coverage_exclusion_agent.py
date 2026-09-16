import time
from typing import Dict, Any, List
from agents.state import ClaimAdjudicationState, ExecutionTrace

class CoverageExclusionAgent:
    """Agent 3: Assesses coverage scope, waiting periods, exclusions, definitions, and financial sublimits."""

    def process(self, state: ClaimAdjudicationState) -> ClaimAdjudicationState:
        t0 = time.time()
        c = state["case_data"]
        facts = state["extracted_facts"]
        evidence = state["retrieved_evidence"]
        
        sum_insured = c.get("sum_insured_inr", 500000)
        continuous_months = c.get("continuous_coverage_months", 0)
        prior_years = c.get("prior_insurer_continuous_years", 0)
        expenses = c.get("expenses_inr", {})
        treatment = c.get("treatment", {})
        prior_policy = c.get("prior_policy", {})

        assessment = {
            "is_covered": True,
            "exclusion_triggered": None,
            "waiting_period_violation": None,
            "sublimit_deductions": [],
            "payable_amount": 0,
            "total_claimed": 0
        }

        # 1. Check Initial 30-day Waiting Period
        # Exception: Continuous coverage under prior Indian insurer (portability)
        effective_coverage_months = continuous_months + (prior_years * 12)
        if continuous_months == 0 and prior_years == 0 and effective_coverage_months < 1:
            assessment["is_covered"] = False
            assessment["waiting_period_violation"] = "30-day Initial Waiting Period (Clause 2)"

        # 2. Check Pre-existing Disease (PED) 48-month Waiting Period
        if facts["pre_existing"]:
            ped_waiting_required_months = 48
            if prior_years > 0 and prior_policy.get("database_and_claim_history_received"):
                ped_waiting_required_months -= (prior_years * 12)
            
            if continuous_months < ped_waiting_required_months:
                assessment["is_covered"] = False
                assessment["waiting_period_violation"] = f"Pre-existing Diseases 48-month Waiting Period (Clause 1). Elapsed: {continuous_months} months."

        # 3. Check Specific Illnesses 1-Year Waiting Period (e.g., Cataract, Joint Replacement, Hysterectomy, Hernia)
        specific_diseases = ["cataract", "joint replacement", "hysterectomy", "hernia", "hydrocele", "fistula", "arthritis", "gout", "stone"]
        diag_lower = facts["diagnosis"].lower()
        proc_lower = facts["procedure"].lower()
        
        is_specific_disease = any(d in diag_lower or d in proc_lower for d in specific_diseases)
        if is_specific_disease:
            required_months = 12
            if prior_years >= 1 and prior_policy.get("database_and_claim_history_received", True):
                required_months = 0 # Portability credit waives 1-year waiting period
            
            if continuous_months < required_months:
                assessment["is_covered"] = False
                assessment["waiting_period_violation"] = f"Specific Illness 1-Year Waiting Period (Clause 3) for {facts['diagnosis']}"

        # 4. Check Explicit Exclusions
        if facts["experimental"]:
            assessment["is_covered"] = False
            assessment["exclusion_triggered"] = "Exclusion 14: Unproven / Experimental Treatment"
        elif "cosmetic" in diag_lower or "cosmetic" in proc_lower:
            assessment["is_covered"] = False
            assessment["exclusion_triggered"] = "Exclusion 5: Cosmetic or Aesthetic Treatment"
        elif facts["treatment_type"] == "domiciliary":
            domiciliary_excluded = ["bronchitis", "asthma", "nephritis", "diarrhoea", "dysentery", "gastro", "diabetes", "epilepsy", "hypertension", "influenza", "cough", "cold", "psychiatric", "tonsillitis"]
            if any(d in diag_lower or d in proc_lower for d in domiciliary_excluded):
                assessment["is_covered"] = False
                assessment["exclusion_triggered"] = f"Exclusion 20: Domiciliary treatment excluded for condition {facts['diagnosis']}"

        # 5. Financial Calculations & Sub-limits
        total_claimed = sum(expenses.values())
        assessment["total_claimed"] = total_claimed

        if assessment["is_covered"]:
            deductions = []
            total_deduction = 0

            # Room Rent Sublimit: 1.0% of Basic Sum Insured per day
            room_claimed = expenses.get("room", 0)
            admission_hours = treatment.get("admission_hours", 24)
            days = max(1, (admission_hours + 23) // 24)
            daily_room_limit = 0.01 * sum_insured
            max_room_allowed = daily_room_limit * days
            
            if room_claimed > max_room_allowed:
                diff = room_claimed - max_room_allowed
                deductions.append({
                    "category": "Room Rent",
                    "claimed": room_claimed,
                    "allowed": max_room_allowed,
                    "deduction": diff,
                    "reason": f"Room rent sublimit of 1.0% SI/day (Rs. {daily_room_limit:.0f}/day for {days} days)"
                })
                total_deduction += diff

            # Doctor / Consultant Fees Sublimit: 25% of Sum Insured
            doc_claimed = expenses.get("doctor_fees", 0)
            doc_limit = 0.25 * sum_insured
            if doc_claimed > doc_limit:
                diff = doc_claimed - doc_limit
                deductions.append({
                    "category": "Doctor Fees",
                    "claimed": doc_claimed,
                    "allowed": doc_limit,
                    "deduction": diff,
                    "reason": f"Surgeon & Consultant fees capped at 25% of Sum Insured (Rs. {doc_limit:.0f})"
                })
                total_deduction += diff

            # Medicines & Diagnostics Sublimit: 40% of Sum Insured
            med_claimed = expenses.get("medicines_diagnostics", 0)
            med_limit = 0.40 * sum_insured
            if med_claimed > med_limit:
                diff = med_claimed - med_limit
                deductions.append({
                    "category": "Medicines & Diagnostics",
                    "claimed": med_claimed,
                    "allowed": med_limit,
                    "deduction": diff,
                    "reason": f"Medicines & diagnostic materials capped at 40% of Sum Insured (Rs. {med_limit:.0f})"
                })
                total_deduction += diff

            # Domiciliary Sublimit: 20% of Basic Sum Insured
            if treatment.get("type") == "domiciliary":
                dom_limit = 0.20 * sum_insured
                if total_claimed > dom_limit:
                    diff = total_claimed - dom_limit
                    deductions.append({
                        "category": "Domiciliary Hospitalization",
                        "claimed": total_claimed,
                        "allowed": dom_limit,
                        "deduction": diff,
                        "reason": f"Domiciliary hospitalization capped at aggregate sublimit of 20% SI (Rs. {dom_limit:.0f})"
                    })
                    total_deduction += diff

            # Ambulance limit: 1.0% of Sum Insured or Rs. 1000 (whichever is less)
            amb_claimed = expenses.get("ambulance", 0)
            amb_limit = min(0.01 * sum_insured, 1000)
            if amb_claimed > amb_limit:
                diff = amb_claimed - amb_limit
                deductions.append({
                    "category": "Ambulance",
                    "claimed": amb_claimed,
                    "allowed": amb_limit,
                    "deduction": diff,
                    "reason": f"Ambulance charges capped at Rs. {amb_limit:.0f}"
                })
                total_deduction += diff

            assessment["sublimit_deductions"] = deductions
            assessment["payable_amount"] = max(0, total_claimed - total_deduction)
        else:
            assessment["payable_amount"] = 0

        state["coverage_assessment"] = assessment
        state["financial_breakdown"] = {
            "total_claimed": total_claimed,
            "payable_amount": assessment["payable_amount"],
            "deductions": assessment.get("sublimit_deductions", [])
        }

        latency = int((time.time() - t0) * 1000)
        trace_entry: ExecutionTrace = {
            "agent": "CoverageExclusionAgent",
            "action": f"Evaluated coverage: Covered={assessment['is_covered']}, Payable=INR {assessment['payable_amount']:.0f}",
            "retrieved_chunks": 0,
            "reranked_chunks": 0,
            "latency_ms": latency
        }
        state["trace"].append(trace_entry)
        return state
