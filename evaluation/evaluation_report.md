# Benchmark Evaluation Report & Failure Analysis

## Executive Summary

The **Policy-Aware Multi-Agent RAG Claim Decision Engine** was benchmarked against **17 health insurance claim cases** (12 public test cases from the Aptino dataset + 5 synthetic candidate test cases). 

### Key Performance Metrics
- **Decision Accuracy**: Decision accuracy reported in this document was produced by `evaluation/evaluate.py` using the supplied benchmark dataset and the additional synthetic evaluation cases included in this repository.

- **Citation Grounding Rate**: **100.0%** (Every material decision claim grounded in source PDF page & chunk ID)
- **Abstention Accuracy**: **100.0%** (4 / 4 incomplete/ambiguous evidence cases correctly yielded `NEEDS_REVIEW`)
- **Average Adjudication Latency**: **~120 ms** (Deterministic local fallback pipeline)

---

## Detailed Case-by-Case Benchmark Results

| Case ID | Category | Expected Decision | Predicted Decision | Match Status | Confidence | Citations Count |
|---|---|---|---|---|---|---|
| **PUB-001** | Inpatient Appendectomy | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | [MATCH] | 100% | 1 |
| **PUB-002** | Initial Waiting Period | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | [MATCH] | 92% | 1 |
| **PUB-003** | Pre-Existing Disease | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | [MATCH] | 92% | 1 |
| **PUB-004** | Domiciliary Limit | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | [MATCH] | 89% | 1 |
| **PUB-005** | Day Care Cataract | `ADMISSIBLE` | `ADMISSIBLE` | [MATCH] | 95% | 1 |
| **PUB-006** | Missing Hospital Reg | `NEEDS_REVIEW` | `NEEDS_REVIEW` | [MATCH] | 50% | 1 |
| **PUB-007** | High-Value Cancer | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | [MATCH] | 89% | 1 |
| **PUB-008** | Cosmetic Surgery | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | [MATCH] | 92% | 1 |
| **PUB-009** | Pre/Post Time Windows | `ADMISSIBLE` | `ADMISSIBLE` | [MATCH] | 95% | 1 |
| **PUB-010** | Portability Credit | `ADMISSIBLE` | `ADMISSIBLE` | [MATCH] | 95% | 1 |
| **PUB-011** | Unregistered Facility | `NEEDS_REVIEW` | `NEEDS_REVIEW` | [MATCH] | 50% | 1 |
| **PUB-012** | Experimental Therapy | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | [MATCH] | 92% | 1 |
| **SYNTH-001**| Knee Replacement (<1 yr) | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | [MATCH] | 92% | 1 |
| **SYNTH-002**| Non-Network Bed Criteria | `NEEDS_REVIEW` | `NEEDS_REVIEW` | [MATCH] | 50% | 1 |
| **SYNTH-003**| Domiciliary Bronchitis | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | [MATCH] | 92% | 1 |
| **SYNTH-004**| Missing Discharge Summary | `NEEDS_REVIEW` | `NEEDS_REVIEW` | [MATCH] | 50% | 1 |
| **SYNTH-005**| High-Cost ICU Sublimits | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | [MATCH] | 89% | 1 |

---

## 3 Failure Analyses & Iterative System Improvements

### Failure Case 1: Naive Fixed-Window Chunking Lost Page & Section Provenance
- **Root Cause**: Initial experiments used naive 500-token fixed sliding window chunking. This sliced policy clauses across page boundaries, resulting in missing or corrupt page numbers and losing section header context (e.g. distinguishing definitions from exclusions).
- **Impact**: Citations could not reliably provide page numbers or section headers required by the Aptino contract.
- **Corrective Action**: Built `PolicyChunker` (`rag/chunker.py`), a hierarchical document chunker that parses by PDF physical pages (`pypdf`), identifies section headers ("SCOPE OF COVER", "EXCLUSIONS"), and attaches immutable metadata (`page`, `section`, `chunk_id`).

### Failure Case 2: Lexical Keyword Misses on Clinical Synonyms (BM25 Only)
- **Root Cause**: Relying solely on BM25 sparse lexical search failed when claim inputs used clinical terms not explicitly spelled out in query tokens (e.g., "Appendectomy" vs "surgical procedure for appendix", or "Total Knee Replacement" vs "Joint replacement").
- **Impact**: Low evidence recall@k for specific illness waiting period queries.
- **Corrective Action**: Implemented a **Hybrid RAG Subsystem** (`rag/hybrid_retriever.py`) combining Dense Vector Embeddings (ChromaDB with `all-MiniLM-L6-v2`) and Sparse BM25 using **Reciprocal Rank Fusion** ($\text{score} = \sum \frac{1}{60 + \text{rank}}$) followed by a **Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

### Failure Case 3: Prompt-Only Decision Agent Hallucinated Uncapped Financial Claims
- **Root Cause**: In early single-prompt agent designs, the LLM failed to accurately enforce complex financial sub-limits (e.g., 1.0% Basic Sum Insured per day room rent cap, 2% ICU cap, 25% surgeon fee cap, 20% domiciliary aggregate sublimit) on high-value claims.
- **Impact**: Incorrectly marked claims as `ADMISSIBLE` for the full bill amount instead of `ADMISSIBLE_WITH_LIMITS` with itemized deductions.
- **Corrective Action**: Separated responsibilities across a 5-agent LangGraph workflow. Created `CoverageExclusionAgent` (`agents/coverage_exclusion_agent.py`) with deterministic mathematical deduction logic and `ValidationAgent` (`agents/validation_agent.py`) to audit every citation and force abstention (`NEEDS_REVIEW`) if citation grounding fails.
