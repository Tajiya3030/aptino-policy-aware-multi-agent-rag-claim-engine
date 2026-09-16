import pytest
from rag.pdf_parser import PolicyPDFParser
from rag.chunker import PolicyChunker
from rag.bm25_retriever import BM25Retriever

def test_pdf_parser_and_chunker():
    parser = PolicyPDFParser("data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf")
    pages = parser.extract_pages()
    assert len(pages) == 17
    
    chunker = PolicyChunker(pages)
    chunks = chunker.chunk_policy()
    assert len(chunks) > 50
    assert "chunk_id" in chunks[0]
    assert "page" in chunks[0]
    assert "section" in chunks[0]

def test_bm25_retriever():
    bm25 = BM25Retriever()
    sample = [
        {"chunk_id": "C1", "content": "Room rent limit is 1% of Sum Insured per day.", "page": 7, "section": "SCOPE", "clause": "Room Rent", "source": "policy.pdf"},
        {"chunk_id": "C2", "content": "Pre-existing disease waiting period is 48 months.", "page": 8, "section": "EXCLUSIONS", "clause": "PED", "source": "policy.pdf"},
        {"chunk_id": "C3", "content": "Ambulance charges limit is 1000 rupees.", "page": 8, "section": "SCOPE", "clause": "Ambulance", "source": "policy.pdf"},
        {"chunk_id": "C4", "content": "Cataract surgery waiting period is 1 year.", "page": 9, "section": "EXCLUSIONS", "clause": "Cataract", "source": "policy.pdf"},
        {"chunk_id": "C5", "content": "Day care treatment under 24 hours.", "page": 2, "section": "DEFINITIONS", "clause": "DayCare", "source": "policy.pdf"}
    ]
    bm25.index_chunks(sample)
    hits = bm25.search("room rent limit")
    assert len(hits) > 0
    assert hits[0]["chunk_id"] == "C1"
