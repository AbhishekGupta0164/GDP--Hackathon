from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List, Dict, Any
from enum import Enum


class Decision(str, Enum):
    ALLOW    = "allow"
    BLOCK    = "block"
    MODIFY   = "modify"
    ESCALATE = "escalate"
    CLARIFY  = "clarify"


class AttackType(str, Enum):
    DIRECT             = "direct"
    SEMANTIC_DISGUISE  = "semantic_disguise"
    ROLEPLAY_JAILBREAK = "roleplay_jailbreak"
    EMOTIONAL_MANIP    = "emotional_manip"
    ENCODED            = "encoded"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"
    EXPERT = "expert"


class RiskLevel(int, Enum):
    SAFE      = 0
    LOW       = 1
    MODERATE  = 2
    HIGH      = 3
    CRITICAL  = 4
    EMERGENCY = 5


class PolicyRule(BaseModel):
    rule_id:                str
    description:            str
    block_keywords:         List[str]
    allow_exceptions:       List[str]
    requires_escalation:    bool
    requires_clarification: bool

    model_config = {"use_enum_values": True}


class ConversationTurn(BaseModel):
    turn:           int
    user_query:     str
    agent_decision: Optional[str] = None
    agent_reason:   Optional[str] = None
    risk_level:     int           = 0
    attack_type:    Optional[str] = None
    flags:          Dict[str, bool] = {}

    model_config = {"use_enum_values": True}


class StateFlags(BaseModel):
    escalation_detected: bool = False
    policy_conflict:     bool = False
    encoded_detected:    bool = False
    emotional_manip:     bool = False
    roleplay_attempt:    bool = False
    late_escalation:     bool = False
    over_blocking:       bool = False
    missed_escalation:   bool = False

    model_config = {"use_enum_values": True}


class AgentAction(BaseModel):
    decision:          str           # string — no Enum, avoids all comparison issues
    reason:            str
    modified_response: Optional[str] = None
    confidence:        float         = 0.8

    @field_validator("decision")
    @classmethod
    def decision_valid(cls, v):
        valid = {"allow", "block", "modify", "escalate", "clarify"}
        if str(v).lower() not in valid:
            raise ValueError("decision must be one of: " + str(valid))
        return str(v).lower()

    @field_validator("reason", mode="before")
    @classmethod
    def reason_not_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("Reason cannot be empty")
        return str(v).strip()

    model_config = {"use_enum_values": True}


class Observation(BaseModel):
    task_id:              str
    turn_number:          int
    max_turns:            int
    current_query:        str
    conversation_history: List[ConversationTurn] = []
    active_policies:      List[PolicyRule]        = []
    risk_level:           int                     = 0
    flags:                StateFlags              = StateFlags()
    context_hint:         Optional[str]           = None
    context:              Optional[Dict[str, Any]] = None

    model_config = {"use_enum_values": True}


class Reward(BaseModel):
    score:     float
    breakdown: Dict[str, float]
    feedback:  str
    penalty:   float = 0.0
    bonus:     float = 0.0
    
    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v):
        return max(0.01, min(0.99, round(float(v), 4)))


class StepResult(BaseModel):
    observation: Observation
    reward:      Reward
    done:        bool
    info:        Dict[str, Any] = {}


class ResetResult(BaseModel):
    session_id:  str
    task_id:     str
    observation: Observation


class StateResult(BaseModel):
    session_id:        str
    task_id:           str
    turn_number:       int
    max_turns:         int
    done:              bool
    risk_level:        int
    flags:             StateFlags
    cumulative_reward: float
    history:           List[ConversationTurn]
    actions_taken:     int


class TaskInfo(BaseModel):
    task_id:       str
    name:          str
    description:   str
    difficulty:    str
    max_turns:     int
    num_scenarios: int
    action_schema: Dict[str, Any]


class GraderResult(BaseModel):
    session_id:      str
    task_id:         str
    final_score:     float
    breakdown:       Dict[str, float]
    feedback:        str
    turns_taken:     int
    flags_triggered: Dict[str, bool]