                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import os
import statistics
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import PROJECT_NAME, VERSION, DESCRIPTION, EXPECTED_BASELINE_SCORES
from app.models import AgentAction, ResetResult, StepResult, StateResult, TaskInfo
from app.env import env_reset, env_step, env_state, env_grader
from app.tasks import list_all_tasks


# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:        str = "easy"
    scenario_index: int = 0


class StepRequest(BaseModel):
    session_id: str
    action:     AgentAction


class GraderRequest(BaseModel):
    session_id: str


# ── Meta Endpoints ────────────────────────────────────────────

# @app.get("/", tags=["meta"])
# def root():
#     return {
#         "environment": PROJECT_NAME,
#         "version":     VERSION,
#         "description": DESCRIPTION,
#         "tasks":       ["easy", "medium", "hard", "expert"],
#         "endpoints":   [
#             "/reset", "/step", "/state",
#             "/tasks", "/grader", "/baseline",
#             "/health", "/ui"
#         ],
#     }

# Fixing  the root to redirect to UI automatically

from fastapi.responses import RedirectResponse

@app.get("/", tags=["meta"])
def root():
    return RedirectResponse(url="/ui")

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "environment": PROJECT_NAME}


# ── Core OpenEnv Endpoints ────────────────────────────────────

@app.post("/reset", response_model=ResetResult, tags=["openenv"])
def reset(request: ResetRequest):
    try:
        return env_reset(request.task_id, request.scenario_index)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult, tags=["openenv"])
def step(request: StepRequest):
    try:
        return env_step(request.session_id, request.action)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResult, tags=["openenv"])
def state(session_id: str = Query(..., description="Session ID from /reset")):
    try:
        return env_state(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", response_model=List[TaskInfo], tags=["openenv"])
def tasks():
    try:
        return list_all_tasks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grader", tags=["openenv"])
def grader(request: GraderRequest):
    try:
        return env_grader(request.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline", tags=["openenv"])
def baseline():
    try:
        from inference import run_baseline
        results = run_baseline()
        overall = round(
            sum(r["mean_score"] for r in results) / len(results), 4
        )
        return {
            "model":        "gpt-4o-mini",
            "results":      results,
            "overall_mean": overall,
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="baseline_inference.py not found or OPENAI_API_KEY not set."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Static UI at /ui ──────────────────────────────────────────

import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.exists(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
    
from fastapi.staticfiles import StaticFiles
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.exists(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")


    