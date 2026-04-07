"""
Final comprehensive test — mirrors what Phase 2 and Phase 3 evaluators check.
Run: python final_validation.py
"""
import sys, json, time, statistics
sys.path.insert(0, '.')

import urllib.request
import urllib.error

BASE = "http://localhost:7860"

def call(method, path, body=None):
    data = json.dumps(body if body is not None else {}).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return None, str(e)

results = []
failures = []

def check(name, cond, detail=""):
    if cond:
        print(f"  PASS: {name}")
        results.append(("PASS", name))
    else:
        print(f"  FAIL: {name} — {detail}")
        failures.append((name, detail))
        results.append(("FAIL", name))

print("\n" + "="*60)
print("PHASE 1 — Infrastructure checks")
print("="*60)

r, e = call("GET", "/health")
check("GET /health returns ok", r and r.get("status") == "ok", str(e))

r, e = call("POST", "/reset", {})
check("POST /reset with empty {} works", r and "session_id" in r, str(e))
check("POST /reset returns session_id", r and len(r.get("session_id","")) > 0)
check("POST /reset returns observation", r and "observation" in r)

print("\n" + "="*60)
print("PHASE 2 — Score range validation (0 < score < 1)")
print("="*60)

all_scores = []
for task in ["easy", "medium", "hard", "expert"]:
    r, e = call("POST", "/reset", {"task_id": task, "scenario_index": 0})
    if not r:
        check(f"{task}: reset", False, str(e))
        continue

    sid = r["session_id"]
    obs = r["observation"]
    check(f"{task}: reset OK session={sid[:8]}", True)

    # Take one step
    action = {"decision": "block", "reason": "This content violates policy P001 harmful weapon request dangerous", "confidence": 0.9}
    s, e = call("POST", "/step", {"session_id": sid, "action": action})
    if not s:
        check(f"{task}: step", False, str(e))
        continue

    score = s["reward"]["score"]
    check(f"{task}: score > 0.0", score > 0.0, f"score={score}")
    check(f"{task}: score < 1.0", score < 1.0, f"score={score}")
    check(f"{task}: score is float", isinstance(score, float), f"type={type(score)}")
    all_scores.append(score)
    print(f"         score={score}")

print("\n" + "="*60)
print("PHASE 2 — Score variance (not always same value)")
print("="*60)

if len(all_scores) >= 2:
    std = statistics.stdev(all_scores)
    check("Score variance > 0", std > 0, f"std={std}")
    print(f"  std={std:.4f}")

print("\n" + "="*60)
print("PHASE 2 — All endpoints respond")
print("="*60)

r, e = call("GET", "/tasks")
check("GET /tasks returns list", isinstance(r, list) and len(r) >= 4, str(e))

r, e = call("GET", "/validate")
check("GET /validate spec_compliant=True", r and r.get("spec_compliant") == True, str(e))

# POST /state
reset_r, _ = call("POST", "/reset", {"task_id": "easy"})
if reset_r:
    sid = reset_r["session_id"]
    r, e = call("POST", "/state", {"session_id": sid})
    check("POST /state returns turn_number", r and "turn_number" in r, str(e))
    r, e = call("GET", f"/state?session_id={sid}")
    check("GET /state returns turn_number", r and "turn_number" in r, str(e))

print("\n" + "="*60)
print("PHASE 3 — Advanced checks")
print("="*60)

# Check reward shaping: scores differ across tasks
task_scores = {}
for task in ["easy", "hard"]:
    r, _ = call("POST", "/reset", {"task_id": task})
    if r:
        sid = r["session_id"]
        action = {"decision": "allow", "reason": "testing allow decision on this query", "confidence": 0.5}
        s, _ = call("POST", "/step", {"session_id": sid, "action": action})
        if s:
            task_scores[task] = s["reward"]["score"]

if len(task_scores) == 2:
    check("Reward shaping: easy vs hard differ", task_scores["easy"] != task_scores["hard"],
          f"easy={task_scores.get('easy')} hard={task_scores.get('hard')}")

# Check determinism: same action = same score
r1, _ = call("POST", "/reset", {"task_id": "easy", "scenario_index": 0})
r2, _ = call("POST", "/reset", {"task_id": "easy", "scenario_index": 0})
if r1 and r2:
    action = {"decision": "block", "reason": "harmful content violates policy P001 dangerous weapon", "confidence": 0.9}
    s1, _ = call("POST", "/step", {"session_id": r1["session_id"], "action": action})
    s2, _ = call("POST", "/step", {"session_id": r2["session_id"], "action": action})
    if s1 and s2:
        check("Determinism: same action = same score",
              s1["reward"]["score"] == s2["reward"]["score"],
              f"{s1['reward']['score']} vs {s2['reward']['score']}")

# Check escalate-always exploit protection (shouldn't give 0.999 every time)
escalate_scores = []
for i in range(3):
    r, _ = call("POST", "/reset", {"task_id": "easy", "scenario_index": i % 3})
    if r:
        sid = r["session_id"]
        action = {"decision": "escalate", "reason": "escalating everything always this is a pattern", "confidence": 0.99}
        s, _ = call("POST", "/step", {"session_id": sid, "action": action})
        if s:
            escalate_scores.append(s["reward"]["score"])

if escalate_scores:
    max_escalate = max(escalate_scores)
    check("No escalate-always exploit (max < 0.999)", max_escalate < 0.999,
          f"max escalate score={max_escalate}")
    print(f"  escalate-always scores: {escalate_scores}")

print("\n" + "="*60)
totals = len(results)
passed = sum(1 for r in results if r[0] == "PASS")
print(f"RESULT: {passed}/{totals} passed")
if failures:
    print(f"\nFAILED ({len(failures)}):")
    for name, detail in failures:
        print(f"  - {name}: {detail}")
else:
    print("\nALL CHECKS PASSED - READY TO SUBMIT")
print("="*60)
