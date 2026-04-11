"""
SafetyGuard X — Final Pre-Deploy Check
Run: python final_check.py
"""
import urllib.request
import json
import sys

BASE = "http://localhost:7860"
PASS = 0
FAIL = 0

def call(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        BASE+path, data=data, method=method,
        headers={"Content-Type":"application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print("  ✅ PASS: " + name)
        PASS += 1
    except Exception as e:
        print("  ❌ FAIL: " + name + " — " + str(e))

        FAIL += 1

print("\n" + "="*55)
print("  SafetyGuard X — Final Pre-Deploy Validation")
print("="*55)

# 1. Health
print("\n[1] Basic Connectivity")
check("GET /health returns 200", lambda: call("GET", "/health"))
check("Response has status=ok", lambda: assert_eq(call("GET", "/health")["status"], "ok"))


# 2. Tasks
print("\n[2] Tasks Endpoint")
tasks = None
def get_tasks():
    global tasks
    tasks = call("GET", "/tasks")
    assert len(tasks) >= 3
check("GET /tasks returns 3+ tasks", get_tasks)
check("Tasks have task_id field", lambda: [t["task_id"] for t in tasks])
check("Tasks have difficulty field", lambda: [t["difficulty"] for t in tasks])
check("4 tasks exist (easy/medium/hard/expert)", lambda: assert_eq(len(tasks), 4))


# 3. Reset
print("\n[3] Reset Endpoint")
sessions = {}
for tid in ["easy", "medium", "hard", "expert"]:
    def test_reset(t=tid):
        global sessions
        r = call("POST", "/reset", {"task_id": t, "scenario_index": 0})
        assert r["session_id"]
        assert r["observation"]["current_query"]
        assert r["observation"]["task_id"] == t
        sessions[t] = r["session_id"]
    check("POST /reset task=" + tid, test_reset)

# 4. Step
print("\n[4] Step Endpoint")
r = call("POST", "/reset", {"task_id": "easy", "scenario_index": 0})
test_sid = r["session_id"]

def test_step_valid():
    action = {"decision":"block","reason":"violates policy P001 harmful weapon request dangerous content","confidence":0.95}
    result = call("POST", "/step", {"session_id": test_sid, "action": action})
    assert 0.0 <= result["reward"]["score"] <= 1.0
    assert "done" in result
    assert "observation" in result
check("POST /step with valid action", test_step_valid)

def test_step_score_range():
    r2 = call("POST", "/reset", {"task_id": "medium", "scenario_index": 0})
    for decision in ["allow", "block", "escalate", "clarify"]:
        action = {"decision": decision, "reason": "policy evaluation for content safety analysis test", "confidence": 0.8}
        result = call("POST", "/step", {"session_id": r2["session_id"], "action": action})
        assert 0.0 <= result["reward"]["score"] <= 1.0
        break
check("Scores always in [0.0, 1.0]", test_step_score_range)

# 5. State
print("\n[5] State Endpoint")
r3 = call("POST", "/reset", {"task_id": "hard", "scenario_index": 0})
sid3 = r3["session_id"]
def test_state():
    s = call("GET", "/state?session_id=" + sid3)
    assert s["task_id"] == "hard"
    assert s["turn_number"] >= 0
    assert "done" in s
check("GET /state returns valid state", test_state)

# 6. Grader
print("\n[6] Grader Endpoint")
r4 = call("POST", "/reset", {"task_id": "easy", "scenario_index": 0})
sid4 = r4["session_id"]
call("POST", "/step", {"session_id": sid4, "action": {"decision":"block","reason":"violates policy P001 harmful weapon dangerous content detected","confidence":0.9}})
def test_grader():
    g = call("POST", "/grader", {"session_id": sid4})
    assert 0.0 <= g["final_score"] <= 1.0
    assert "breakdown" in g
    assert "feedback" in g
check("POST /grader returns valid score", test_grader)

# 7. Graders not always same score
print("\n[7] Grader Produces Varied Scores")
scores_collected = []
for scenario in range(3):
    r_t = call("POST", "/reset", {"task_id": "easy", "scenario_index": scenario})
    s_t = r_t["session_id"]
    for dec in ["block", "allow"]:
        call("POST", "/step", {"session_id": s_t, "action": {"decision": dec, "reason": "policy evaluation test for grader variance check", "confidence": 0.8}})
        g_t = call("POST", "/grader", {"session_id": s_t})
        scores_collected.append(g_t["final_score"])
        break

def test_score_variance():
    assert len(set(scores_collected)) > 1, "Grader returns same score always — DISQUALIFICATION RISK"
check("Grader produces varied scores (not always same)", test_score_variance)

# 8. Baseline inference
print("\n[8] Inference Script")
import subprocess
def test_inference():
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True, text=True, timeout=120
    )
    assert "OVERALL MEAN" in result.stdout, "inference.py did not complete: " + result.stderr[:200]

check("inference.py runs and produces scores", test_inference)

# 9. Required files exist
print("\n[9] Required Files")
import os
required = ["inference.py", "Dockerfile", "openenv.yaml", "requirements.txt", "README.md", "pyproject.toml", "baseline_inference.py"]
for f in required:
    check("File exists: " + f, lambda fn=f: assert_true(os.path.exists(fn)))


# 10. Dockerfile has port 7860
print("\n[10] Dockerfile Check")
def test_dockerfile():
    with open("Dockerfile") as f:
        content = f.read()
    assert "7860" in content
    assert "FROM python" in content
check("Dockerfile has port 7860", test_dockerfile)

# 11. openenv.yaml valid
print("\n[11] openenv.yaml Check")
def test_openenv_yaml():
    import yaml
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert "name" in data
    assert "tasks" in data
    assert len(data["tasks"]) >= 3
try:
    import yaml
    check("openenv.yaml has name + 3+ tasks", test_openenv_yaml)
except ImportError:
    print("  ⚠️  SKIP: openenv.yaml (pyyaml not installed)")

# ── Helpers ──────────────────────────────────────────────────
def assert_eq(a, b):
    assert a == b, str(a) + " != " + str(b)

def assert_true(v, msg="Assertion failed"):
    assert v, msg

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*55)
print("  RESULTS: " + str(PASS) + " passed | " + str(FAIL) + " failed")
if FAIL == 0:
    print("  🎉 ALL CHECKS PASSED — READY TO DEPLOY!")
else:
    print("  ⚠️  Fix " + str(FAIL) + " issues before deploying")

print("="*55 + "\n")