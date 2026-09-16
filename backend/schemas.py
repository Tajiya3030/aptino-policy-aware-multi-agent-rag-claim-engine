from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CitationSchema(BaseModel):
    claim: str = Field(..., description="Policy-supported statement")
    source: str = Field(..., description="Policy source document name")
    page: int = Field(..., description="Physical PDF page number")
    section: str = Field(..., description="Policy section header")
    chunk_id: str = Field(..., description="Unique chunk identifier")

class ValidationSchema(BaseModel):
    status: str = Field(..., description="Validation audit outcome: PASS or FAIL")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of unsupported claim statements")

class RetrievalMetadataSchema(BaseModel):
    dense_hits: int = Field(..., description="Number of dense vector search hits")
    bm25_hits: int = Field(..., description="Number of BM25 lexical search hits")
    rrf_candidates: int = Field(..., description="Number of Reciprocal Rank Fusion candidate chunks")
    reranked_top_k: int = Field(..., description="Number of reranked top chunks passed to agents")
    top_score: float = Field(..., description="Highest reranker score among top chunks")

class ExecutionTraceSchema(BaseModel):
    agent: str = Field(..., description="Specialized agent name")
    action: str = Field(..., description="Action performed by agent")
    retrieved_chunks: int = Field(default=0, description="Retrieved chunk count")
    reranked_chunks: int = Field(default=0, description="Reranked chunk count")
    latency_ms: int = Field(..., description="Action latency in milliseconds")

class AdjudicationResponseSchema(BaseModel):
    case_id: str = Field(..., description="Unique claim case identifier")
    decision: str = Field(..., description="Adjudication decision status (ADMISSIBLE, ADMISSIBLE_WITH_LIMITS, PARTIALLY_ADMISSIBLE, NOT_ADMISSIBLE, NEEDS_REVIEW)")
    confidence: float = Field(..., description="Evidence-grounded multi-signal confidence score")
    key_findings: List[str] = Field(default_factory=list, description="Key adjudication findings")
    applicable_limits: List[str] = Field(default_factory=list, description="Category-specific limits & financial deductions")
    missing_evidence: List[str] = Field(default_factory=list, description="List of missing evidence items when NEEDS_REVIEW")
    citations: List[CitationSchema] = Field(default_factory=list, description="Source policy citations")
    retrieval_metadata: RetrievalMetadataSchema = Field(..., description="Retrieval metadata metrics")
    validation: ValidationSchema = Field(..., description="Validation audit result")
    trace: List[ExecutionTraceSchema] = Field(default_factory=list, description="Agent execution trace")

class HealthCheckSchema(BaseModel):
    status: str
    app_name: str
    version: str
    vector_store_status: str
    llm_configured: bool
