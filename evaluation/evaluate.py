import os
import json
import time
import sys
from typing import List, Dict, Any
from agents.workflow import ClaimAdjudicationWorkflow
from rag.indexer import PolicyIndexer

# Ground truth expected decisions for 17 benchmark cases
EXPECTED_DECISIONS = {
    "PUB-001": "ADMISSIBLE_WITH_LIMITS",
    "PUB-002": "NOT_ADMISSIBLE",
    "PUB-003": "NOT_ADMISSIBLE",
    "PUB-004": "ADMISSIBLE_WITH_LIMITS",
    "PUB-005": "ADMISSIBLE",
    "PUB-006": "NEEDS_REVIEW",
    "PUB-007": "ADMISSIBLE_WITH_LIMITS",
    "PUB-008": "NOT_ADMISSIBLE",
    "PUB-009": "ADMISSIBLE",
    "PUB-010": "ADMISSIBLE",
    "PUB-011": "NEEDS_REVIEW",
    "PUB-012": "NOT_ADMISSIBLE",
    "SYNTH-001": "NOT_ADMISSIBLE",
    "SYNTH-002": "NEEDS_REVIEW",
    "SYNTH-003": "NOT_ADMISSIBLE",
    "SYNTH-004": "NEEDS_REVIEW",
    "SYNTH-005": "ADMISSIBLE_WITH_LIMITS"
}

def run_evaluation():
    print("===================================================================")
    print("      POLICY-AWARE MULTI-AGENT RAG CLAIM ENGINE EVALUATION        ")
    print("===================================================================")
    
    indexer = PolicyIndexer()
    indexer.build_index()
    workflow = ClaimAdjudicationWorkflow(indexer=indexer)
    
    with open("evaluation/test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    correct_decisions = 0
    total_cases = len(test_cases)
    
    total_citations = 0
    grounded_citations = 0
    
    abstention_target_cases = [c_id for c_id, exp in EXPECTED_DECISIONS.items() if exp == "NEEDS_REVIEW"]
    correct_abstentions = 0
    
    results = []

    for case in test_cases:
        c_id = case["case_id"]
        expected = EXPECTED_DECISIONS.get(c_id, "UNKNOWN")
        
        t0 = time.time()
        output = workflow.run(case)
        elapsed_ms = int((time.time() - t0) * 1000)
        
        predicted = output["decision"]
        is_match = (predicted == expected)
        if is_match:
            correct_decisions += 1

        if expected == "NEEDS_REVIEW" and predicted == "NEEDS_REVIEW":
            correct_abstentions += 1

        cites = output.get("citations", [])
        total_citations += len(cites)
        if output.get("validation", {}).get("status") == "PASS":
            grounded_citations += len(cites)

        results.append({
            "case_id": c_id,
            "expected": expected,
            "predicted": predicted,
            "match": "MATCH" if is_match else "MISMATCH",
            "confidence": output.get("confidence", 0.0),
            "citations_count": len(cites),
            "latency_ms": elapsed_ms
        })

    accuracy = (correct_decisions / total_cases) * 100.0
    grounding_rate = (grounded_citations / max(1, total_citations)) * 100.0
    abstention_acc = (correct_abstentions / len(abstention_target_cases)) * 100.0

    print("\n-------------------------------------------------------------------")
    print(f"Total Benchmark Cases Evaluated : {total_cases}")
    print(f"Decision Accuracy              : {accuracy:.2f}% ({correct_decisions}/{total_cases})")
    print(f"Citation Grounding Rate        : {grounding_rate:.2f}% ({grounded_citations}/{total_citations})")
    print(f"Abstention Accuracy           : {abstention_acc:.2f}% ({correct_abstentions}/{len(abstention_target_cases)})")
    print("-------------------------------------------------------------------\n")

    print(f"{'CASE ID':<12} | {'EXPECTED':<22} | {'PREDICTED':<22} | {'RESULT':<10} | {'LATENCY'}")
    print("-" * 80)
    for r in results:
        status_str = "[MATCH]" if r["match"] == "MATCH" else "[MISMATCH]"
        print(f"{r['case_id']:<12} | {r['expected']:<22} | {r['predicted']:<22} | {status_str:<10} | {r['latency_ms']}ms")

    report_data = {
        "summary": {
            "total_cases": total_cases,
            "decision_accuracy_pct": accuracy,
            "citation_grounding_pct": grounding_rate,
            "abstention_accuracy_pct": abstention_acc,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "case_results": results
    }

    with open("evaluation/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data

if __name__ == "__main__":
    run_evaluation()
