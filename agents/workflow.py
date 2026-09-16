from typing import Dict, Any
from langgraph.graph import StateGraph, END
from agents.state import ClaimAdjudicationState
from agents.case_analysis_agent import CaseAnalysisAgent
from agents.policy_evidence_agent import PolicyEvidenceAgent
from agents.coverage_exclusion_agent import CoverageExclusionAgent
from agents.decision_agent import DecisionAgent
from agents.validation_agent import ValidationAgent
from rag.indexer import PolicyIndexer

class ClaimAdjudicationWorkflow:
    """Orchestrates multi-agent claim adjudication state transitions in LangGraph."""

    def __init__(self, indexer: PolicyIndexer = None):
        if indexer is None:
            indexer = PolicyIndexer()
            indexer.build_index()
        self.indexer = indexer

        self.case_analysis_agent = CaseAnalysisAgent()
        self.policy_evidence_agent = PolicyEvidenceAgent(indexer)
        self.coverage_exclusion_agent = CoverageExclusionAgent()
        self.decision_agent = DecisionAgent()
        self.validation_agent = ValidationAgent()

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ClaimAdjudicationState)

        # Add agent nodes
        workflow.add_node("case_analysis", self.case_analysis_agent.process)
        workflow.add_node("policy_evidence", self.policy_evidence_agent.process)
        workflow.add_node("coverage_exclusion", self.coverage_exclusion_agent.process)
        workflow.add_node("decision", self.decision_agent.process)
        workflow.add_node("validation", self.validation_agent.process)

        # Set entry point
        workflow.set_entry_point("case_analysis")

        # Define linear transitions
        workflow.add_edge("case_analysis", "policy_evidence")
        workflow.add_edge("policy_evidence", "coverage_exclusion")
        workflow.add_edge("coverage_exclusion", "decision")
        workflow.add_edge("decision", "validation")

        # Define conditional routing from validation node
        def check_validation(state: ClaimAdjudicationState):
            val = state.get("validation", {})
            retries = state.get("retry_count", 0)
            if val.get("status") == "FAIL" and retries < 1:
                state["retry_count"] = retries + 1
                return "policy_evidence"
            return END

        workflow.add_conditional_edges("validation", check_validation, {
            "policy_evidence": "policy_evidence",
            END: END
        })

        return workflow.compile()

    def run(self, claim_case_json: Dict[str, Any]) -> Dict[str, Any]:
        initial_state: ClaimAdjudicationState = {
            "case_data": claim_case_json,
            "case_id": claim_case_json.get("case_id", "UNKNOWN"),
            "extracted_facts": {},
            "investigation_plan": [],
            "decision_dimensions": [],
            "missing_fields": [],
            "retrieved_evidence": [],
            "retrieval_metadata": {
                "dense_hits": 0,
                "bm25_hits": 0,
                "rrf_candidates": 0,
                "reranked_top_k": 0,
                "top_score": 0.0
            },
            "coverage_assessment": {},
            "financial_breakdown": {},
            "draft_decision": "NEEDS_REVIEW",
            "confidence": 0.50,
            "key_findings": [],
            "applicable_limits": [],
            "missing_evidence": [],
            "citations": [],
            "validation": {"status": "PASS", "unsupported_claims": []},
            "retry_count": 0,
            "trace": []
        }

        final_state = self.graph.invoke(initial_state)
        
        # Build machine-readable response matching exact contract specification
        return {
            "case_id": final_state["case_id"],
            "decision": final_state["draft_decision"],
            "confidence": final_state["confidence"],
            "key_findings": final_state["key_findings"],
            "applicable_limits": final_state["applicable_limits"],
            "missing_evidence": final_state["missing_evidence"],
            "citations": final_state["citations"],
            "retrieval_metadata": final_state["retrieval_metadata"],
            "validation": final_state["validation"],
            "trace": final_state["trace"]
        }

if __name__ == "__main__":
    import json
    wf = ClaimAdjudicationWorkflow()
    with open("data/candidate_data/public_test_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    if cases:
        out = wf.run(cases[0])
        print("Sample Run Output:\n", json.dumps(out, indent=2))
