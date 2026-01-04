from __future__ import annotations

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

from src.api_stub.runner import run_turn
from src.app.settings import load_settings
from src.db.mongo import connect_mongo
from src.db.repositories import SessionRepo, TurnRepo

app = FastAPI(
    title="Tesco Creative Builder API",
    description="AI-powered retail media creative generation API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TurnRequest(BaseModel):
    user_text: str = Field(..., description="User prompt for creative generation")
    session_id: Optional[str] = Field(None, description="Existing session ID (creates new if None)")
    attachments: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Image attachments")
    ui_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="UI context (formats, etc.)")
    title_if_new: Optional[str] = Field("New Session", description="Title for new session")
    session_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session configuration")

class TurnResponse(BaseModel):
    session_id: str
    turn_id: str
    turn_index: int
    compliance_result: str
    summary: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str

# Initialize DB connection (lazy)
_settings = None
_handles = None

def get_db_handles():
    global _settings, _handles
    if _handles is None:
        _settings = load_settings()
        _handles = connect_mongo(_settings.mongo_uri, _settings.mongo_db)
    return _handles

# Endpoints
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Tesco Creative Builder API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        handles = get_db_handles()
        # Test DB connection
        handles["sessions"].find_one({"_id": "health_check"})
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database=db_status
    )

@app.post("/api/v1/turns", response_model=TurnResponse, tags=["Turns"])
async def create_turn(request: TurnRequest):
    """
    Create a new creative generation turn.
    
    This endpoint triggers the full creative generation pipeline:
    - Copy validation
    - Layout planning
    - Image generation (Gemini)
    - Compliance checking
    - Export optimization
    
    Returns the turn result with artifacts and compliance status.
    """
    try:
        result = run_turn(
            session_id=request.session_id,
            user_text=request.user_text,
            attachments=request.attachments or [],
            ui_context=request.ui_context or {},
            title_if_new=request.title_if_new or "New Session",
            session_config_if_new=request.session_config or {},
        )
        return TurnResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Turn generation failed: {str(e)}")

@app.get("/api/v1/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Get session details"""
    try:
        handles = get_db_handles()
        session_repo = SessionRepo(handles["sessions"])
        session = session_repo.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sessions/{session_id}/turns", tags=["Sessions"])
async def get_session_turns(session_id: str, limit: int = 50, skip: int = 0):
    """Get all turns for a session"""
    try:
        handles = get_db_handles()
        turn_repo = TurnRepo(handles["turns"])
        turns = list(turn_repo.get_turns_by_session(session_id, limit=limit, skip=skip))
        return {
            "session_id": session_id,
            "count": len(turns),
            "turns": turns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/turns/{turn_id}", tags=["Turns"])
async def get_turn(turn_id: str):
    """Get specific turn details"""
    try:
        handles = get_db_handles()
        turn_repo = TurnRepo(handles["turns"])
        turn = turn_repo.get_turn(turn_id)
        if not turn:
            raise HTTPException(status_code=404, detail="Turn not found")
        return turn
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/artifacts/{session_id}/{turn_id}/{filename}", tags=["Artifacts"])
async def get_artifact(session_id: str, turn_id: str, filename: str):
    """Download generated artifact image"""
    artifact_path = f"artifacts/{session_id}/{turn_id}/{filename}"
    if not os.path.exists(artifact_path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(artifact_path, media_type="image/png")
