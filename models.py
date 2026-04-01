# models.py
from pydantic import BaseModel
from typing import List, Optional, Literal

class SafetyAction(BaseModel):
    decision: Literal["allow", "block", "modify", "escalate", "clarify"]
    reason: str
    modified_response: Optional[str] = None
    confidence: float = 0.8


class SafetyObservation(BaseModel):
    current_query: str
    task_id: str
    turn_number: int
    max_turns: int
    risk_level: int
    active_policies: List[dict] = []
    conversation_history: List[dict] = []
    done: bool = False


class SafetyState(BaseModel):
    session_id: str
    episode_id: str
    step_count: int
    total_reward: float = 0.0
    done: bool = False


# Export for OpenEnv
Action = SafetyAction
Observation = SafetyObservation
State = SafetyState