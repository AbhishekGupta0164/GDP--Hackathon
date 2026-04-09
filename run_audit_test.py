"""
Full audit test — simulates what the OpenEnv checker does.
Recursively checks every float in every API response for boundary values.
"""

import urllib.request
import json
import math
import sys

BASE = "http://localhost:7860"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=15).read().decode())


def check_no_boundary_floats(obj, path="root"):
    """Recursively assert no float equals exactly 0.0 or 1.0."""
    if isinstance(obj, float):
        assert 0.0 < obj < 1.0, f"BOUNDARY FLOAT at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_boundary_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_no_boundary_floats(v, f"{path}[{i}]")


errors = []
tasks = ["easy", "medium", "hard", "expert"]

# ── Test 1: empty body reset ──────────────────────────────────
try:
    r = call("POST", "/reset", {})
    print("Test 1 PASS: empty reset =>", r["session_id"][:8])
    check_no_boundary_floats(r, "reset/empty")
except Exception as e:
    errors.append(f"empty-reset: {e}")
    print(f"Test 1 FAIL: {e}")


# ── Per-task tests ────────────────────────────────────────────
for task in tasks:
    try:
        # Reset
        r = call("POST", "/reset", {"task_id": task, "scenario_index": 0})
        sid = r["session_id"]
        check_no_boundary_floats(r, f"reset/{task}")

        # Step with block
        s = call("POST", "/step", {
            "session_id": sid,
            "action": {
                "decision": "block",
                "reason": "violates policy P001 harmful content detected",
                "confidence": 0.9,
            },
        })
        check_no_boundary_floats(s, f"step/{task}")
        score = s["reward"]["score"]
        assert 0.0 < score < 1.0, f"Score boundary in {task}: {score}"
        print(f"Test {task} PASS: score={score:.4f}")

        # Grader
        g = call("POST", "/grader", {"session_id": sid})
        check_no_boundary_floats(g, f"grader/{task}")
        fs = g["final_score"]
        print(f"Grader {task} PASS: final={fs:.4f}")

    except AssertionError as e:
        errors.append(str(e))
        print(f"FAIL: {e}")
    except Exception as e:
        errors.append(str(e))
        print(f"ERROR on {task}: {e}")


# ── Validate endpoint ─────────────────────────────────────────
try:
    v = call("GET", "/validate")
    print("Validate PASS:", v.get("spec_compliant"))
except Exception as e:
    errors.append(f"validate: {e}")
    print(f"Validate FAIL: {e}")


# ── Health check ──────────────────────────────────────────────
try:
    h = call("GET", "/health")
    print("Health PASS:", h.get("status"))
except Exception as e:
    errors.append(f"health: {e}")
    print(f"Health FAIL: {e}")


# ── Baseline endpoint (informational — OpenEnv checker does NOT call /baseline) ──
try:
    b = call("POST", "/baseline", {})
    check_no_boundary_floats(b, "baseline")
    om = b.get("overall_mean")
    print(f"Baseline PASS: overall_mean={om}")
    for res in b.get("results", []):
        assert "std_score" not in res, f"std_score still in baseline result for {res.get('task_id')}"
except Exception as e:
    # /baseline runs 12 live episodes — it may time out in local test.
    # This does NOT affect the OpenEnv submission checker which only tests
    # /reset, /step, /grader, /validate, /health.
    msg = str(e)
    if "timed out" in msg or "timeout" in msg.lower():
        print(f"Baseline SKIP (timeout — expected for live-episode endpoint): {msg}")
    else:
        errors.append(f"baseline: {e}")
        print(f"Baseline FAIL: {e}")



# ── Summary ───────────────────────────────────────────────────
print()
if errors:
    print(f"{len(errors)} ERRORS FOUND — fix before pushing:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL TESTS PASS — safe to push and resubmit")
