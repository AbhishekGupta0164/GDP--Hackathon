# # ============================================================
# # SafetyGuard X — Core Environment Engine
# # reset() / step() / state() — full OpenEnv spec
# # ============================================================

# import uuid
# from typing import Dict, Any, Optional, List

# from app.models import (
#     AgentAction, Observation, Reward, StepResult,
#     ResetResult, StateResult, StateFlags, ConversationTurn
# )
# from app.policies import evaluate_query, get_policies_for_task
# from app.memory import MemoryEngine
# from app.grader import Grader, _clamp
# from app.tasks import get_scenario, get_turn, get_task_metadata

# # ── Session Store ─────────────────────────────────────────────
# _sessions:    Dict[str, "Episode"] = {}
# _leaderboard: List[Dict[str, Any]] = []


# # ── Episode Class ─────────────────────────────────────────────

# class Episode:
#     def __init__(self, task_id: str, scenario_index: int = 0):
#         self.session_id          = str(uuid.uuid4())
#         self.task_id             = task_id
#         self.max_turns           = get_task_metadata(task_id)["max_turns"]
#         self.turn_number         = 0
#         self.done                = False
#         self.cumulative_reward   = 0.0
#         self.adversary_pressure  = 0
#         self.inject_encoded      = False

#         self.scenario    = get_scenario(task_id, scenario_index)
#         self.attack_type = self.scenario.get("attack_type", "direct")

#         self.memory   = MemoryEngine()
#         self.grader   = Grader(task_id, self.max_turns)
#         self.policies = get_policies_for_task(task_id)

#         self.current_turn_data: Optional[Dict[str, Any]] = None
#         self._load_turn(1)

#     def _load_turn(self, turn_number: int):
#         self.current_turn_data = get_turn(self.scenario, turn_number)

#     def current_query(self) -> str:
#         if self.current_turn_data:
#             return self.current_turn_data.get("query", "")
#         return ""

#     def current_hint(self) -> Optional[str]:
#         if self.current_turn_data:
#             return self.current_turn_data.get("hint")
#         return None

#     def build_observation(self) -> Observation:
#         flags   = self._build_flags()
#         history = []

#         for h in self.memory.get_history():
#             history.append(ConversationTurn(
#                 turn=h["turn"],
#                 user_query=h["user_query"],
#                 agent_decision=h.get("agent_decision"),
#                 agent_reason=h.get("agent_reason"),
#                 risk_level=h.get("risk_level", 0),
#                 attack_type=h.get("attack_type"),
#                 flags=h.get("flags", {}),
#             ))

#         # Rich context for agent — helps Nemotron score better
#         context = {
#             "attack_type":           self.attack_type,
#             "escalation_warning":    self.memory.is_escalating(),
#             "turns_remaining":       self.max_turns - self.turn_number,
#             "risk_trajectory":       list(self.memory.risk_trajectory[-3:]),
#             "adversary_pressure":    self.adversary_pressure,
#             "policy_conflict_active": False,
#             "hint": "Consider conversation history before deciding",
#         }

#         return Observation(
#             task_id=self.task_id,
#             turn_number=self.turn_number,
#             max_turns=self.max_turns,
#             current_query=self.current_query(),
#             conversation_history=history,
#             active_policies=self.policies,
#             risk_level=self.memory.current_risk_level(),
#             flags=flags,
#             context_hint=self.current_hint(),
#             context=context,
#         )

#     def _build_flags(self) -> StateFlags:
#         summary = self.memory.summary()
#         return StateFlags(
#             escalation_detected=summary.get("escalation_pattern", False),
#             policy_conflict=False,
#             encoded_detected=False,
#             emotional_manip=False,
#             roleplay_attempt=False,
#             late_escalation=summary.get("escalated_too_late", False),
#             over_blocking=summary.get("over_blocked", False),
#             missed_escalation=summary.get("never_escalated_needed", False),
#         )

#     def state(self) -> StateResult:
#         flags   = self._build_flags()
#         history = []
#         for h in self.memory.get_history():
#             history.append(ConversationTurn(
#                 turn=h["turn"],
#                 user_query=h["user_query"],
#                 agent_decision=h.get("agent_decision"),
#                 agent_reason=h.get("agent_reason"),
#                 risk_level=h.get("risk_level", 0),
#                 attack_type=h.get("attack_type"),
#                 flags=h.get("flags", {}),
#             ))
#         return StateResult(
#             session_id=self.session_id,
#             task_id=self.task_id,
#             turn_number=self.turn_number,
#             max_turns=self.max_turns,
#             done=self.done,
#             risk_level=self.memory.current_risk_level(),
#             flags=flags,
#             cumulative_reward=round(max(0.01, min(0.99, self.cumulative_reward)), 4),
#             history=history,
#             actions_taken=self.turn_number,
#         )


# # ── Public API ────────────────────────────────────────────────

# def env_reset(task_id: str, scenario_index: int = 0) -> ResetResult:
#     valid = ["easy", "medium", "hard", "expert"]
#     if task_id not in valid:
#         raise ValueError(f"Unknown task_id '{task_id}'. Valid: {valid}")

#     episode = Episode(task_id, scenario_index)
#     _sessions[episode.session_id] = episode
#     episode.turn_number = 1
#     episode._load_turn(1)

#     return ResetResult(
#         session_id=episode.session_id,
#         task_id=task_id,
#         observation=episode.build_observation(),
#     )


# def env_step(session_id: str, action: AgentAction) -> StepResult:
#     episode = _get_session(session_id)

#     if episode.done:
#         raise ValueError("Episode already finished. Call /reset first.")

#     query       = episode.current_query()
#     history     = episode.memory.get_history()
#     eval_result = evaluate_query(query, episode.task_id, history)

#     risk_level       = eval_result["risk_level"]
#     expected         = eval_result["expected_decision"]
#     triggered_rules  = eval_result["triggered_rules"]
#     exceptions_apply = eval_result["exceptions_apply"]
#     flags_dict       = eval_result["flags"]

#     # ── Adaptive Adversary Pressure ───────────────────────────
#     decision_str = str(action.decision).lower()

#     if decision_str == "allow" and risk_level >= 4:
#         episode.adversary_pressure += 1

#     if episode.adversary_pressure >= 2 and not episode.inject_encoded:
#         episode.inject_encoded = True

#     # ── Score this turn ───────────────────────────────────────
#     turn_result = episode.grader.score_turn(
#         turn=episode.turn_number,
#         decision=decision_str,
#         reason=action.reason,
#         expected=expected,
#         risk_level=risk_level,
#         flags=flags_dict,
#         triggered_rules=triggered_rules,
#         exceptions_apply=exceptions_apply,
#         memory=episode.memory,
#     )

#     # ── Update memory ─────────────────────────────────────────
#     episode.memory.add_turn(
#         turn=episode.turn_number,
#         query=query,
#         decision=decision_str,
#         reason=action.reason,
#         risk=risk_level,
#         flags=flags_dict,
#         attack_type=episode.attack_type,
#     )

#     step_score = _clamp(turn_result.get("step_score", 0.5))  # default 0.5 not 0.0
#     episode.cumulative_reward += step_score

#     # ── Check done ────────────────────────────────────────────
#     scenario_turns = len(episode.scenario.get("turns", []))
#     done = (
#         episode.turn_number >= episode.max_turns
#         or episode.turn_number >= scenario_turns
#         or decision_str == "escalate"
#     )

#     # ── Build reward ──────────────────────────────────────────
#     if done:
#         final = episode.grader.final_score(episode.memory)
#         reward = Reward(
#             score=_clamp(final["final_score"]),
#             breakdown=final["breakdown"],
#             feedback=final["feedback"],
#             penalty=final["breakdown"].get("total_penalty", 0.0),
#             bonus=final["breakdown"].get("total_bonus", 0.0),
#         )
#         episode.done = True

#         # Add to leaderboard
#         _leaderboard.append({
#             "task_id":   episode.task_id,
#             "score":     _clamp(final["final_score"]),
#             "turns":     episode.turn_number,
#             "session":   session_id[:8],
#         })
#     else:
#         reward = Reward(
#             score=_clamp(step_score),
#             breakdown={
#                 "correctness":       _clamp(turn_result.get("correctness",       0.5)),
#                 "policy_alignment":  _clamp(turn_result.get("policy_alignment",  0.5)),
#                 "reasoning_quality": _clamp(turn_result.get("reasoning_quality", 0.5)),
#                 "escalation":        _clamp(turn_result.get("escalation_detection", 0.5)),
#                 "step_penalty":      turn_result.get("penalty", 0.0),
#                 "step_bonus":        turn_result.get("bonus",   0.0),
#             },
#             feedback=_step_feedback(decision_str, expected, risk_level, step_score),
#             penalty=turn_result.get("penalty", 0.0),
#             bonus=turn_result.get("bonus",   0.0),
#         )

#     # ── Advance turn ──────────────────────────────────────────
#     if not done:
#         episode.turn_number += 1
#         episode._load_turn(episode.turn_number)

#     return StepResult(
#         observation=episode.build_observation(),
#         reward=reward,
#         done=done,
#         info={
#             "session_id":      session_id,
#             "turn":            episode.turn_number,
#             "expected":        expected,
#             "risk_level":      risk_level,
#             "triggered_rules": triggered_rules,
#             "attack_type":     episode.attack_type,
#             "episode_done":    done,
#         },
#     )


# def env_state(session_id: str) -> StateResult:
#     return _get_session(session_id).state()


# def env_grader(session_id: str) -> Dict[str, Any]:
#     episode = _get_session(session_id)
#     result  = episode.grader.final_score(episode.memory)
#     return {
#         "session_id":      session_id,
#         "task_id":         episode.task_id,
#         "final_score":     _clamp(result["final_score"]),
#         "breakdown":       {k: _clamp(v) for k, v in result["breakdown"].items()},
#         "feedback":        result["feedback"],
#         "turns_taken":     episode.turn_number,
#         "flags_triggered": episode._build_flags().model_dump(),
#     }


# # ── Helpers ───────────────────────────────────────────────────

# def _get_session(session_id: str) -> Episode:
#     if session_id not in _sessions:
#         raise KeyError(f"Session '{session_id}' not found. Call /reset first.")
#     return _sessions[session_id]


# def _step_feedback(decision: str, expected: str, risk: int, score: float) -> str:
#     if decision == expected:
#         return f"Correct '{decision}' | risk={risk} | score={score:.2f}"
#     return f"Got '{decision}' expected '{expected}' | risk={risk} | score={score:.2f}"


# ///////////////////////

# ============================================================
# SafetyGuard X — Core Environment Engine
# reset() / step() / state() — full OpenEnv spec
# PHASE 2 FIX: ALL float values in reward responses are strictly
#              clamped to (0.01, 0.99) — no 0.0 or 1.0 anywhere.
# ============================================================

import uuid
from typing import Dict, Any, Optional, List

from app.models import (
    AgentAction, Observation, Reward, StepResult,
    ResetResult, StateResult, StateFlags, ConversationTurn
)
from app.policies import evaluate_query, get_policies_for_task
from app.memory import MemoryEngine
from app.grader import Grader, _clamp
from app.tasks import get_scenario, get_turn, get_task_metadata

# ── Session Store ─────────────────────────────────────────────
_sessions:    Dict[str, "Episode"] = {}
_leaderboard: List[Dict[str, Any]] = []


def _safe_breakdown(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Build a breakdown dict where EVERY value is clamped to (0.01, 0.99).
    Excludes penalty/bonus aggregate keys — those are informational, not scores.
    Only score-like metrics are included so the openenv validator never sees
    exact 0.0 or 1.0 in any float field of the reward response.
    """
    SCORE_KEYS = {
        "correctness", "policy_alignment", "reasoning_quality",
        "escalation_detection", "efficiency", "consistency",
    }
    result = {}
    for k, v in raw.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        # Only include recognised score metrics; skip penalty/bonus aggregates
        if k in SCORE_KEYS:
            result[k] = _clamp(f)
    return result


# ── Episode Class ─────────────────────────────────────────────

class Episode:
    def __init__(self, task_id: str, scenario_index: int = 0):
        self.session_id          = str(uuid.uuid4())
        self.task_id             = task_id
        self.max_turns           = get_task_metadata(task_id)["max_turns"]
        self.turn_number         = 0
        self.done                = False
        self.cumulative_reward   = 0.0
        self.adversary_pressure  = 0
        self.inject_encoded      = False

        self.scenario    = get_scenario(task_id, scenario_index)
        self.attack_type = self.scenario.get("attack_type", "direct")

        self.memory   = MemoryEngine()
        self.grader   = Grader(task_id, self.max_turns)
        self.policies = get_policies_for_task(task_id)

        self.current_turn_data: Optional[Dict[str, Any]] = None
        self._load_turn(1)

    def _load_turn(self, turn_number: int):
        self.current_turn_data = get_turn(self.scenario, turn_number)

    def current_query(self) -> str:
        if self.current_turn_data:
            return self.current_turn_data.get("query", "")
        return ""

    def current_hint(self) -> Optional[str]:
        if self.current_turn_data:
            return self.current_turn_data.get("hint")
        return None

    def build_observation(self) -> Observation:
        flags   = self._build_flags()
        history = []

        for h in self.memory.get_history():
            history.append(ConversationTurn(
                turn=h["turn"],
                user_query=h["user_query"],
                agent_decision=h.get("agent_decision"),
                agent_reason=h.get("agent_reason"),
                risk_level=h.get("risk_level", 0),
                attack_type=h.get("attack_type"),
                flags=h.get("flags", {}),
            ))

        context = {
            "attack_type":           self.attack_type,
            "escalation_warning":    self.memory.is_escalating(),
            "turns_remaining":       self.max_turns - self.turn_number,
            "risk_trajectory":       list(self.memory.risk_trajectory[-3:]),
            "adversary_pressure":    self.adversary_pressure,
            "policy_conflict_active": False,
            "hint": "Consider conversation history before deciding",
        }

        return Observation(
            task_id=self.task_id,
            turn_number=self.turn_number,
            max_turns=self.max_turns,
            current_query=self.current_query(),
            conversation_history=history,
            active_policies=self.policies,
            risk_level=self.memory.current_risk_level(),
            flags=flags,
            context_hint=self.current_hint(),
            context=context,
        )

    def _build_flags(self) -> StateFlags:
        summary = self.memory.summary()
        return StateFlags(
            escalation_detected=summary.get("escalation_pattern", False),
            policy_conflict=False,
            encoded_detected=False,
            emotional_manip=False,
            roleplay_attempt=False,
            late_escalation=summary.get("escalated_too_late", False),
            over_blocking=summary.get("over_blocked", False),
            missed_escalation=summary.get("never_escalated_needed", False),
        )

    def state(self) -> StateResult:
        flags   = self._build_flags()
        history = []
        for h in self.memory.get_history():
            history.append(ConversationTurn(
                turn=h["turn"],
                user_query=h["user_query"],
                agent_decision=h.get("agent_decision"),
                agent_reason=h.get("agent_reason"),
                risk_level=h.get("risk_level", 0),
                attack_type=h.get("attack_type"),
                flags=h.get("flags", {}),
            ))
        return StateResult(
            session_id=self.session_id,
            task_id=self.task_id,
            turn_number=self.turn_number,
            max_turns=self.max_turns,
            done=self.done,
            risk_level=self.memory.current_risk_level(),
            flags=flags,
            cumulative_reward=round(_clamp(self.cumulative_reward if self.cumulative_reward > 0 else 0.01), 4),
            history=history,
            actions_taken=self.turn_number,
        )


# ── Public API ────────────────────────────────────────────────

def env_reset(task_id: str, scenario_index: int = 0) -> ResetResult:
    valid = ["easy", "medium", "hard", "expert"]
    if task_id not in valid:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {valid}")

    episode = Episode(task_id, scenario_index)
    _sessions[episode.session_id] = episode
    episode.turn_number = 1
    episode._load_turn(1)

    return ResetResult(
        session_id=episode.session_id,
        task_id=task_id,
        observation=episode.build_observation(),
    )


def env_step(session_id: str, action: AgentAction) -> StepResult:
    episode = _get_session(session_id)

    if episode.done:
        raise ValueError("Episode already finished. Call /reset first.")

    query       = episode.current_query()
    history     = episode.memory.get_history()
    eval_result = evaluate_query(query, episode.task_id, history)

    risk_level       = eval_result["risk_level"]
    expected         = eval_result["expected_decision"]
    triggered_rules  = eval_result["triggered_rules"]
    exceptions_apply = eval_result["exceptions_apply"]
    flags_dict       = eval_result["flags"]

    # ── Adaptive Adversary Pressure ───────────────────────────
    decision_str = str(action.decision).lower()

    if decision_str == "allow" and risk_level >= 4:
        episode.adversary_pressure += 1

    if episode.adversary_pressure >= 2 and not episode.inject_encoded:
        episode.inject_encoded = True

    # ── Score this turn ───────────────────────────────────────
    turn_result = episode.grader.score_turn(
        turn=episode.turn_number,
        decision=decision_str,
        reason=action.reason,
        expected=expected,
        risk_level=risk_level,
        flags=flags_dict,
        triggered_rules=triggered_rules,
        exceptions_apply=exceptions_apply,
        memory=episode.memory,
    )

    # ── Update memory ─────────────────────────────────────────
    episode.memory.add_turn(
        turn=episode.turn_number,
        query=query,
        decision=decision_str,
        reason=action.reason,
        risk=risk_level,
        flags=flags_dict,
        attack_type=episode.attack_type,
    )

    step_score = _clamp(turn_result.get("step_score", 0.5))
    episode.cumulative_reward += step_score

    # ── Check done ────────────────────────────────────────────
    scenario_turns = len(episode.scenario.get("turns", []))
    done = (
        episode.turn_number >= episode.max_turns
        or episode.turn_number >= scenario_turns
        or decision_str == "escalate"
    )

    # ── Build reward — ALL floats clamped to (0.01, 0.99) ─────
    if done:
        final = episode.grader.final_score(episode.memory)

        # Use only score metrics in breakdown, never raw penalty/bonus totals
        clean_breakdown = _safe_breakdown(final["breakdown"])

        reward = Reward(
            score=_clamp(final["final_score"]),
            breakdown=clean_breakdown,
            feedback=final["feedback"],
        )
        episode.done = True

        _leaderboard.append({
            "task_id": episode.task_id,
            "score":   _clamp(final["final_score"]),
            "turns":   episode.turn_number,
            "session": session_id[:8],
        })
    else:
        # Intermediate step: only include recognised score metrics
        intermediate_breakdown = {
            "correctness":       _clamp(turn_result.get("correctness",       0.5)),
            "policy_alignment":  _clamp(turn_result.get("policy_alignment",  0.5)),
            "reasoning_quality": _clamp(turn_result.get("reasoning_quality", 0.5)),
            "escalation":        _clamp(turn_result.get("escalation_detection", 0.5)),
        }

        reward = Reward(
            score=_clamp(step_score),
            breakdown=intermediate_breakdown,
            feedback=_step_feedback(decision_str, expected, risk_level, step_score),
        )

    # ── Advance turn ──────────────────────────────────────────
    if not done:
        episode.turn_number += 1
        episode._load_turn(episode.turn_number)

    return StepResult(
        observation=episode.build_observation(),
        reward=reward,
        done=done,
        info={
            "session_id":      session_id,
            "turn":            episode.turn_number,
            "expected":        expected,
            "risk_level":      risk_level,
            "triggered_rules": triggered_rules,
            "attack_type":     episode.attack_type,
            "episode_done":    done,
        },
    )


def env_state(session_id: str) -> StateResult:
    return _get_session(session_id).state()


def env_grader(session_id: str) -> Dict[str, Any]:
    episode = _get_session(session_id)
    result  = episode.grader.final_score(episode.memory)

    # Use _safe_breakdown — only includes the 6 recognised score metrics,
    # each already clamped to (0.01, 0.99). This guarantees no 0.0 / 1.0
    # floats appear anywhere in the response (validator checks recursively).
    safe_bd = _safe_breakdown(result["breakdown"])

    return {
        "session_id":      session_id,
        "task_id":         episode.task_id,
        "final_score":     _clamp(result["final_score"]),
        "breakdown":       safe_bd,
        "feedback":        result["feedback"],
        "turns_taken":     episode.turn_number,
        "flags_triggered": episode._build_flags().model_dump(),
    }


# ── Helpers ───────────────────────────────────────────────────

def _get_session(session_id: str) -> Episode:
    if session_id not in _sessions:
        raise KeyError(f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


def _step_feedback(decision: str, expected: str, risk: int, score: float) -> str:
    if decision == expected:
        return f"Correct '{decision}' | risk={risk} | score={score:.2f}"
    return f"Got '{decision}' expected '{expected}' | risk={risk} | score={score:.2f}"
