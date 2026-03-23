import sys
from pathlib import Path
from typing import List, Optional
import time

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from pipeline.hybrid_pipeline import HybridPipeline

app = FastAPI(title="Vietnamese Legal QA API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline lazily
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing HybridPipeline for API...")
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        _pipeline = HybridPipeline(config_path)
    return _pipeline

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "web_session"
    verbose: bool = True

class QueryResponse(BaseModel):
    answer: str
    route_used: str
    confidence: float
    router_reasoning: str
    stage2_invoked: bool
    sources: List[str]
    latency_ms: float
    is_ambiguous: bool

@app.get("/")
async def root():
    return {"message": "Vietnamese Legal QA API is running", "status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_legal(request: QueryRequest):
    try:
        pipeline = get_pipeline()
        logger.info(f"API Query: {request.query}")
        
        response = pipeline.query(
            query=request.query,
            session_id=request.session_id,
            verbose=request.verbose
        )
        
        return QueryResponse(
            answer=response.answer,
            route_used=response.route_used,
            confidence=response.confidence,
            router_reasoning=response.router_reasoning,
            stage2_invoked=response.stage2_invoked,
            sources=response.sources,
            latency_ms=response.latency_ms,
            is_ambiguous=response.is_ambiguous
        )
    except Exception as e:
        logger.exception("API Error:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
