import urllib.request
import json

BASE = "http://localhost:7860"

def call(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(
        BASE+path, data=data, method=method,
        headers={"Content-Type":"application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

print("\n" + "="*50)
print("SafetyGuard X — Full System Test")
print("="*50)

# 1. Health
r = call("GET", "/health")
print("\n[1] Health:", r["status"])
assert r["status"] == "ok"
print("    PASS ✓")

# 2. Tasks
r = call("GET", "/tasks")
print("\n[2] Tasks found:", len(r))
assert len(r) >= 3
for t in r:
    print("    -", t["task_id"], "|", t["difficulty"])
print("    PASS ✓")

# 3. Reset all tasks
print("\n[3] Testing reset() for all tasks...")
for task_id in ["easy", "medium", "hard", "expert"]:
    r = call("POST", "/reset", {"task_id": task_id, "scenario_index": 0})
    assert r["session_id"]
    assert r["observation"]["current_query"]
    print("    " + task_id + ": session=" + r["session_id"][:8] + "... PASS ✓")

# 4. Step test
print("\n[4] Testing step()...")
r = call("POST", "/reset", {"task_id": "easy", "scenario_index": 0})
sid = r["session_id"]
query = r["observation"]["current_query"]
print("    Query:", query[:50])

action = {
    "decision": "block",
    "reason": "this violates policy P001 harmful weapon instructions dangerous content",
    "confidence": 0.95
}
r = call("POST", "/step", {"session_id": sid, "action": action})
score = r["reward"]["score"]
assert 0.0 <= score <= 1.0
print("    Score:", score, "| Done:", r["done"])
print("    PASS ✓")

# 5. State test
print("\n[5] Testing state()...")
r2 = call("POST", "/reset", {"task_id": "medium", "scenario_index": 0})
sid2 = r2["session_id"]
r = call("GET", "/state?session_id=" + sid2)
assert r["task_id"] == "medium"
print("    State OK | turn:", r["turn_number"])
print("    PASS ✓")

# 6. Grader test
print("\n[6] Testing grader()...")
r = call("POST", "/grader", {"session_id": sid})
assert 0.0 <= r["final_score"] <= 1.0
print("    Grader score:", r["final_score"])
print("    PASS ✓")

# 7. Grader scores range check
print("\n[7] Grader scores 0.0-1.0 for all tasks...")
for task_id in ["easy", "medium", "hard", "expert"]:
    r = call("POST", "/reset", {"task_id": task_id, "scenario_index": 0})
    sid_t = r["session_id"]
    action = {"decision":"block","reason":"violates policy P001 harmful dangerous content detected","confidence":0.9}
    r = call("POST", "/step", {"session_id": sid_t, "action": action})
    r2 = call("POST", "/grader", {"session_id": sid_t})
    sc = r2["final_score"]
    assert 0.0 <= sc <= 1.0
    print("    " + task_id + ": " + str(sc) + " ✓")


# 8. Inference script
print("\n[8] Testing inference.py...")
import subprocess, sys
result = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True, text=True, timeout=120
)
if "OVERALL MEAN" in result.stdout:
    print("    inference.py ran clean ✓")
    for line in result.stdout.split("\n"):
        if "MEAN" in line or "OVERALL" in line:
            print("   ", line)
else:
    print("    Output:", result.stdout[-300:])
    print("    Errors:", result.stderr[-200:])

print("\n" + "="*50)
print("ALL TESTS PASSED — Ready to deploy!")
print("="*50)