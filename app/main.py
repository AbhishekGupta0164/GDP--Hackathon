# ============================================================
# SafetyGuard X — FastAPI Server
# ============================================================

import os, statistics
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.grader import _clamp
from app.config import PROJECT_NAME, VERSION, DESCRIPTION
from app.models import AgentAction, ResetResult, StepResult, StateResult, TaskInfo
from app.env import env_reset, env_step, env_state, env_grader, _leaderboard
from app.tasks import list_all_tasks
from app.trainer import run_training
from app.exporter import exporter

# ── App ───────────────────────────────────────────────────────
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



@app.get("/", include_in_schema=False)
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        paths = [
            os.path.join(os.path.dirname(__file__), "static", "index.html"),
            "app/static/index.html",
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
    return {
        "environment":  PROJECT_NAME,
        "version":      VERSION,
        "description":  DESCRIPTION,
        "tasks":        ["easy", "medium", "hard", "expert"],
        "endpoints":    ["/reset", "/step", "/state", "/tasks",
                        "/grader", "/baseline", "/health", "/ui",
                        "/validate", "/leaderboard", "/train", "/export_dataset"],
        "ui":           "/ui",
        "docs":         "/docs",
    }

@app.get("/health", tags=["meta"])
def health():
    return {
        "status":      "ok",
        "environment": PROJECT_NAME,
        "version":     VERSION
    }


@app.get("/metadata", tags=["openenv"])
def get_metadata():
    return {
        "name":        PROJECT_NAME,
        "description": DESCRIPTION,
        "version":     VERSION,
        "author":      "SafetyGuard X Team",
        "license":     "MIT"
    }


@app.get("/schema", tags=["openenv"])
def get_schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["allow", "block", "modify", "escalate", "clarify"]},
                "reason": {"type": "string", "min_length": 10},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["decision", "reason"]
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "turn_number": {"type": "integer"},
                "current_query": {"type": "string"},
                "risk_level": {"type": "integer"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "done": {"type": "boolean"},
                "turn_number": {"type": "integer"}
            }
        }
    }


@app.post("/mcp", tags=["openenv"])
def post_mcp():
    return {
        "jsonrpc": "2.0",
        "result": {
            "status": "active",
            "capabilities": ["multi_turn", "grader"]
        },
        "id": 1
    }


# ── Core OpenEnv Endpoints ────────────────────────────────────

# Add these new endpoints (keep all existing ones)

POST /api/audit          # Upload CSV + run full fairness audit
GET  /api/tasks          # List available bias tasks
POST /api/train          # Start PPO training (same as before)
GET  /api/metrics/{run}  # Get fairness metrics for a run
POST /api/mitigate       # Apply mitigation strategy
GET  /api/report/{run}   # Download PDF/HTML report
POST /api/gemini/explain # Get Gemini narrative explanation
POST /api/counterfactual # Counterfactual analysis

@app.post("/reset", tags=["openenv"])
async def reset(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        task_id        = str(body.get("task_id", "easy"))
        scenario_index = int(body.get("scenario_index", 0))
        return env_reset(task_id, scenario_index)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", tags=["openenv"])
async def step(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        session_id = body.get("session_id", "")
        action_data = body.get("action", {"decision": "block", "reason": "default safety block", "confidence": 0.8})
        if not session_id:
            raise HTTPException(status_code=422, detail="session_id required")
        action = AgentAction(**action_data)
        return env_step(session_id, action)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/state", tags=["openenv"])
async def state_post(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        session_id = body.get("session_id", "")
        if not session_id:
            raise HTTPException(status_code=422, detail="session_id required")
        return env_state(session_id)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", tags=["openenv"])
async def state_get(session_id: str = Query(...)):
    try:
        return env_state(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks", tags=["openenv"])
def tasks():
    try:
        return list_all_tasks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grader", tags=["openenv"])
async def grader(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        session_id = body.get("session_id", "")
        if not session_id:
            raise HTTPException(status_code=422, detail="session_id required")
        return env_grader(session_id)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/baseline", tags=["openenv"])
def baseline():
    try:
        from baseline_inference import run_baseline
        results = run_baseline()
        for r in results:
            r["mean_score"] = _clamp(r["mean_score"])
        overall = _clamp(statistics.mean(r["mean_score"] for r in results) if results else 0.5)
        return {"model": "mock_agent", "results": results, "overall_mean": overall}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/validate", tags=["openenv"])
def validate():
    tasks_list = list_all_tasks()
    return {
        "name":               PROJECT_NAME,
        "version":            VERSION,
        "spec_compliant":     True,
        "tasks":              [t.model_dump() for t in tasks_list],
        "tasks_with_graders": [t.task_id for t in tasks_list],
        "has_autograder":     True,
        "graders": {
            t.task_id: t.grader for t in tasks_list
        },
        "endpoints":          ["/reset","/step","/state","/tasks","/grader","/baseline","/leaderboard", "/metadata", "/schema", "/mcp", "/train", "/export_dataset"],
        "reward_range":       [0.01, 0.99],
        "deterministic":      True,
        "multi_turn":         True,
        "capabilities":       ["multi_turn", "grader"],
        "attack_types":       ["direct","semantic_disguise","roleplay_jailbreak","emotional_manip","encoded"],
    }

@app.get("/leaderboard", tags=["openenv"])
def leaderboard():
    from app.env import _leaderboard
    sorted_lb = sorted(_leaderboard, key=lambda x: x["score"], reverse=True)
    avg = _clamp(statistics.mean(x["score"] for x in _leaderboard) if _leaderboard else 0.5)
    return {
        "top_scores":     sorted_lb[:10],
        "total_episodes": len(_leaderboard),
        "average_score":  avg,
    }

# ── Global Status ─────────────────────────────────────────────
TRAINING_STATUS = {"active": False, "last_episodes": 0, "last_task": None, "current_ep": 0}

@app.post("/train", tags=["v3.0"])
async def train_policy(request: Request, background_tasks: BackgroundTasks):
    if TRAINING_STATUS["active"]:
        return {"success": False, "message": "Training already in progress."}
        
    try:
        body = await request.json()
        episodes = int(body.get("episodes", 100))
        task_id = body.get("task", "expert")
        
        TRAINING_STATUS["active"] = True
        TRAINING_STATUS["last_episodes"] = episodes
        TRAINING_STATUS["last_task"] = task_id
        TRAINING_STATUS["current_ep"] = 0
        
        def run_n_update():
             try:
                 def on_ep(n):
                     TRAINING_STATUS["current_ep"] = n
                 run_training(episodes=episodes, task_id=task_id, on_episode_end=on_ep)
                 TRAINING_STATUS["current_ep"] = episodes
             finally:
                 TRAINING_STATUS["active"] = False

        background_tasks.add_task(run_n_update)
        
        return {
            "success": True,
            "message": f"Started training pipeline for {episodes} episodes in background.",
            "status_url": "/train/status"
        }
    except Exception as e:
        TRAINING_STATUS["active"] = False
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/status", tags=["v3.0"])
def get_train_status():
    return TRAINING_STATUS

@app.get("/export_dataset", tags=["v3.0"])
def export_dataset():
    try:
        from app.env import _sessions
        all_episodes = []
        for sess in list(_sessions.values()):
            all_episodes.append({
                "session_id": sess.session_id,
                "history": sess.memory.get_history()
            })
        
        if not all_episodes:
            raise HTTPException(status_code=400, detail="No episodes available to export.")
            
        ds_path = exporter.export_episodes(all_episodes)
        return FileResponse(
            path=ds_path, 
            filename=os.path.basename(ds_path),
            media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Static UI ─────────────────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")

# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)