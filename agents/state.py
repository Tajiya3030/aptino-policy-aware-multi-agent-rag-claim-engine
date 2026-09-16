from typing import List, Dict, Any, TypedDict, Optional

class Citation(TypedDict):
    claim: str
    source: str
    page: int
    section: str
    chunk_id: str

class ValidationResult(TypedDict):
    status: str  # "PASS" or "FAIL"
    unsupported_claims: List[str]

class ExecutionTrace(TypedDict):
    agent: str
    action: str
    retrieved_chunks: int
    reranked_chunks: int
    latency_ms: int

class RetrievalMetadata(TypedDict):
    dense_hits: int
    bm25_hits: int
    rrf_candidates: int
    reranked_top_k: int
    top_score: float

class ClaimAdjudicationState(TypedDict):
    case_data: Dict[str, Any]
    case_id: str
    extracted_facts: Dict[str, Any]
    investigation_plan: List[str]
    decision_dimensions: List[str]
    missing_fields: List[str]
    retrieved_evidence: List[Dict[str, Any]]
    retrieval_metadata: RetrievalMetadata
    coverage_assessment: Dict[str, Any]
    financial_breakdown: Dict[str, Any]
    draft_decision: str
    confidence: float
    key_findings: List[str]
    applicable_limits: List[str]
    missing_evidence: List[str]
    citations: List[Citation]
    validation: ValidationResult
    retry_count: int
    trace: List[ExecutionTrace]
