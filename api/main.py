from pathlib import Path
from typing import List, Optional
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from pipeline.hybrid_pipeline import HybridPipeline
from api.auth import router as auth_router, get_current_user

app = FastAPI(title="Vietnamese Legal QA API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")

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
    k: Optional[int] = 3
    verbose: bool = True
    is_clarify_answer: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    route_used: str
    confidence: float
    router_reasoning: str
    stage2_invoked: bool
    stage2_override: bool
    sources: List[str]
    latency_ms: float
    is_ambiguous: bool
    resolved_query: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Vietnamese Legal QA API is running", "status": "ok"}

@app.post("/query")
async def query(request: QueryRequest, username: str = Depends(get_current_user)):
    try:
        pipeline = get_pipeline()
        logger.info(f"API Query: {request.query}")
        
        response = pipeline.query(
            query=request.query,
            session_id=request.session_id,
            verbose=request.verbose,
            username=username
        )
        
        return QueryResponse(
            answer=response.answer,
            route_used=response.route_used,
            confidence=response.confidence,
            router_reasoning=response.router_reasoning,
            stage2_invoked=response.stage2_invoked,
            stage2_override=response.stage2_override,
            sources=response.sources,
            latency_ms=response.latency_ms,
            is_ambiguous=response.is_ambiguous,
            resolved_query=getattr(response, "resolved_query", None)
        )
    except Exception as e:
        logger.exception("API Error:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_stream")
async def query_stream(request: QueryRequest, username: str = Depends(get_current_user)):
    try:
        pipeline = get_pipeline()
        logger.info(f"API Stream Query: {request.query}")
        
        def generate():
            for chunk in pipeline.query_stream(
                query=request.query,
                session_id=request.session_id,
                verbose=request.verbose,
                username=username,
                force_route="dense_retrieval" if request.is_clarify_answer else None
            ):
                yield chunk
                
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.exception("API Stream Error:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_all_sessions(username: str = Depends(get_current_user)):
    try:
        pipeline = get_pipeline()
        return {"sessions": pipeline.conversation_manager.get_all_sessions(username)}
    except Exception as e:
        logger.exception("API Error:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str, username: str = Depends(get_current_user)):
    try:
        pipeline = get_pipeline()
        
        # Check if session belongs to user
        sessions = pipeline.conversation_manager.get_all_sessions(username)
        turns = pipeline.conversation_manager.get_history(session_id)
        
        # If the session exists (has turns) but doesn't belong to the user, deny access
        if turns and not any(s['session_id'] == session_id for s in sessions):
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
            
        return {"session_id": session_id, "history": turns}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("API Error:")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, username: str = Depends(get_current_user)):
    try:
        pipeline = get_pipeline()
        pipeline.conversation_manager.clear_session(session_id, username)
        return {"status": "success", "message": f"Session {session_id} deleted."}
    except Exception as e:
        logger.exception("API Error:")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}/truncate")
async def truncate_session(session_id: str, keep_turns: int, username: str = Depends(get_current_user)):
    try:
        pipeline = get_pipeline()
        sessions = pipeline.conversation_manager.get_all_sessions(username)
        if not any(s['session_id'] == session_id for s in sessions):
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
            
        pipeline.conversation_manager.truncate_session(session_id, keep_turns)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("API Error:")
        raise HTTPException(status_code=500, detail=str(e))

class SessionUpdate(BaseModel):
    title: str

@app.put("/sessions/{session_id}")
async def update_session(session_id: str, payload: SessionUpdate, username: str = Depends(get_current_user)):
    try:
        pipeline = get_pipeline()
        pipeline.conversation_manager.update_session_title(session_id, payload.title, username)
        return {"status": "success", "message": f"Session {session_id} renamed."}
    except Exception as e:
        logger.exception("API Error:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
