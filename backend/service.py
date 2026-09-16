from typing import Dict, Any
from agents.workflow import ClaimAdjudicationWorkflow
from rag.indexer import PolicyIndexer

class AdjudicationService:
    """Service layer orchestrating policy indexing and LangGraph agent workflow execution."""

    def __init__(self):
        print("[AdjudicationService] Initializing RAG Policy Indexer & LangGraph Workflow...")
        self.indexer = PolicyIndexer()
        self.indexer.build_index()
        self.workflow = ClaimAdjudicationWorkflow(indexer=self.indexer)

    def analyze_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.workflow.run(claim_data)

# Global service singleton instance
adjudication_service = None

def get_adjudication_service() -> AdjudicationService:
    global adjudication_service
    if adjudication_service is None:
        adjudication_service = AdjudicationService()
    return adjudication_service
