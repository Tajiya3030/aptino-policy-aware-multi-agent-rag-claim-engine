import os
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.schemas import AdjudicationResponseSchema, HealthCheckSchema
from backend.service import get_adjudication_service, AdjudicationService

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade Policy-Aware Multi-Agent RAG Claim Decision Engine adhering strictly to Aptino specifications."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckSchema, tags=["Health"])
def health_check(service: AdjudicationService = Depends(get_adjudication_service)):
    is_indexed = service.indexer.chunks is not None and len(service.indexer.chunks) > 0
    return {
        "status": "healthy" if is_indexed else "degraded",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "vector_store_status": f"indexed_{len(service.indexer.chunks)}_chunks" if is_indexed else "unindexed",
        "llm_configured": bool(settings.gemini_api_key)
    }

@app.post("/analyze", response_model=AdjudicationResponseSchema, tags=["Adjudication"])
def analyze_claim(claim_data: dict, service: AdjudicationService = Depends(get_adjudication_service)):
    if not claim_data or not isinstance(claim_data, dict):
        raise HTTPException(status_code=400, detail="Invalid input JSON. Must be a valid claim case object.")
    
    try:
        response = service.analyze_claim(claim_data)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adjudication pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
