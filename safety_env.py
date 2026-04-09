import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

def _clamp_score(v: float) -> float:
    """Clamp score-like values to strict open interval (0.01, 0.99)."""
    try:
        f = float(v)
        if f <= 0.01:
            return 0.01
        if f >= 0.99:
            return 0.99
        return f
    except (TypeError, ValueError):
        return 0.5

class SafetyGuardXAction:
    def __init__(self, decision: str, reason: str, confidence: float = 0.8, modified_response: Optional[str] = None):
        self.decision = decision
        self.reason = reason
        self.confidence = confidence
        self.modified_response = modified_response

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "decision": self.decision,
            "reason": self.reason,
            "confidence": self.confidence,
        }
        if self.modified_response:
            d["modified_response"] = self.modified_response
        return d

class SafetyGuardXObservation:
    def __init__(self, data: Dict[str, Any]):
        self.task_id = data.get("task_id")
        self.turn_number = data.get("turn_number")
        self.max_turns = data.get("max_turns")
        self.current_query = data.get("current_query")
        self.conversation_history = data.get("conversation_history", [])
        self.active_policies = data.get("active_policies", [])
        self.risk_level = data.get("risk_level", 0)
        self.flags = data.get("flags", {})
        self.context_hint = data.get("context_hint")
        self.context = data.get("context")
        self.done = data.get("done", False) # Often useful here

class SafetyGuardXResult:
    def __init__(self, observation_data: Dict[str, Any], reward_data: Optional[Dict[str, Any]] = None, done: bool = False):
        self.observation = SafetyGuardXObservation(observation_data)
        self.reward = _clamp_score(reward_data.get("score", 0.5)) if reward_data else 0.5
        self.done = done

class SafetyGuardXEnv:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self.task_id: Optional[str] = None

    @classmethod
    async def from_docker_image(cls, image_name: str = None):
        # In this specific repo, we assume the server is running or started separately
        # But we return the instance to match the sample's usage
        return cls()

    async def reset(self, task_id: str = "easy", scenario_index: int = 0):
        url = f"{self.base_url}/reset"
        body = {"task_id": task_id, "scenario_index": scenario_index}
        data = self._call("POST", url, body)
        
        self.session_id = data["session_id"]
        self.task_id = data["task_id"]
        
        # We need to preserve the session state
        return SafetyGuardXResult(data["observation"])

    async def step(self, action: SafetyGuardXAction):
        if not self.session_id:
            raise RuntimeError("Env not reset")
            
        url = f"{self.base_url}/step"
        body = {
            "session_id": self.session_id,
            "action": action.to_dict()
        }
        data = self._call("POST", url, body)
        
        obs_data = data["observation"]
        reward_data = data["reward"]
        done = data["done"]
        
        return SafetyGuardXResult(obs_data, reward_data, done)

    async def get_grader_score(self) -> float:
        if not self.session_id:
            return 0.5
        url = f"{self.base_url}/grader"
        body = {"session_id": self.session_id}
        data = self._call("POST", url, body)
        return _clamp_score(data.get("final_score", 0.5))

    async def close(self):
        # Cleanup if needed
        pass

    def _call(self, method: str, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            print(f"[ERROR] Environment call failed: {e}")
            raise
