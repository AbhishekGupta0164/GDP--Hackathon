# import sys
# sys.path.insert(0, 'C:\\XProMetaAI')

# from app.grader import Grader
# from app.memory import MemoryEngine


# def test_grader_correct_decision():
#     m = MemoryEngine()
#     g = Grader("easy", max_turns=3)
#     result = g.score_turn(
#         turn=1,
#         decision="block",
#         reason="violates policy P001 harmful weapon request dangerous",
#         expected="block",
#         risk_level=5,
#         flags={
#             "escalation_needed": True,
#             "encoded_detected": False,
#             "emotional_manip": False,
#             "policy_conflict": False,
#             "roleplay_attempt": False,
#         },
#         triggered_rules=["P001"],
#         exceptions_apply=False,
#         memory=m,
#     )
#     assert result["step_score"] >= 0.5
#     print("PASS: correct decision scores >= 0.5 | score=" + str(result["step_score"]))


# def test_grader_wrong_decision_low_score():
#     m = MemoryEngine()
#     g = Grader("easy", max_turns=3)
#     result = g.score_turn(
#         turn=1,
#         decision="allow",
#         reason="seems fine",
#         expected="block",
#         risk_level=5,
#         flags={
#             "escalation_needed": True,
#             "encoded_detected": False,
#             "emotional_manip": False,
#             "policy_conflict": False,
#             "roleplay_attempt": False,
#         },
#         triggered_rules=["P001"],
#         exceptions_apply=False,
#         memory=m,
#     )
#     assert result["step_score"] <= 0.3
#     print("PASS: wrong decision scores <= 0.3 | score=" + str(result["step_score"]))


# def test_grader_score_range():
#     m = MemoryEngine()
#     g = Grader("hard", max_turns=7)
#     for decision in ["allow", "block", "escalate", "clarify", "modify"]:
#         result = g.score_turn(
#             turn=1,
#             decision=decision,
#             reason="policy evaluation based on content risk analysis",
#             expected="block",
#             risk_level=3,
#             flags={
#                 "escalation_needed": False,
#                 "encoded_detected": False,
#                 "emotional_manip": False,
#                 "policy_conflict": False,
#                 "roleplay_attempt": False,
#             },
#             triggered_rules=["P003"],
#             exceptions_apply=False,
#             memory=m,
#         )
#         assert 0.0 <= result["step_score"] <= 1.0
#     print("PASS: all scores in 0.0-1.0 range")


# def test_final_score_range():
#     m = MemoryEngine()
#     g = Grader("medium", max_turns=5)
#     g.score_turn(
#         turn=1,
#         decision="block",
#         reason="harmful content violates policy P001",
#         expected="block",
#         risk_level=4,
#         flags={
#             "escalation_needed": False,
#             "encoded_detected": False,
#             "emotional_manip": False,
#             "policy_conflict": False,
#             "roleplay_attempt": False,
#         },
#         triggered_rules=["P001"],
#         exceptions_apply=False,
#         memory=m,
#     )
#     m.add_turn(1, "bomb query", "block", "harmful", 4, {})
#     final = g.final_score(m)
#     assert 0.0 <= final["final_score"] <= 1.0
#     assert "breakdown" in final
#     assert "feedback" in final
#     print("PASS: final score valid | score=" + str(final["final_score"]))


# def test_missed_escalation_penalty():
#     m = MemoryEngine()
#     g = Grader("expert", max_turns=10)
#     # Two turns with high risk but agent allowed both
#     for turn in [1, 2]:
#         g.score_turn(
#             turn=turn,
#             decision="allow",
#             reason="seems ok",
#             expected="escalate",
#             risk_level=5,
#             flags={
#                 "escalation_needed": True,
#                 "encoded_detected": False,
#                 "emotional_manip": False,
#                 "policy_conflict": False,
#                 "roleplay_attempt": False,
#             },
#             triggered_rules=["P001", "P008"],
#             exceptions_apply=False,
#             memory=m,
#         )
#         m.add_turn(turn, "dangerous query", "allow", "seems ok", 5, {})
#     final = g.final_score(m)
#     assert final["breakdown"]["total_penalty"] > 0
#     print("PASS: missed escalation penalty applied | penalty=" +
#           str(final["breakdown"]["total_penalty"]))


# if __name__ == "__main__":
#     test_grader_correct_decision()
#     test_grader_wrong_decision_low_score()
#     test_grader_score_range()
#     test_final_score_range()
#     test_missed_escalation_penalty()
#     print("\nALL GRADER TESTS PASSED")



import sys
sys.path.insert(0, '.')
from app.grader import _clamp, _SCORE_MIN, _SCORE_MAX, Grader
from app.memory import MemoryEngine

# Test _clamp boundaries
assert _clamp(0.0) == 0.001, 'FAIL: clamp(0.0) != 0.001'
assert _clamp(1.0) == 0.999, 'FAIL: clamp(1.0) != 0.999'
assert _clamp(0.5) == 0.5, 'FAIL: clamp(0.5) != 0.5'
assert 0.0 < _clamp(-999) < 1.0
assert 0.0 < _clamp(999) < 1.0
print('_clamp: OK')

# Run a grader episode and check every score is in open (0,1)
g = Grader('easy', 3)
m = MemoryEngine()
for turn in range(1, 4):
    result = g.score_turn(
        turn=turn, decision='block',
        reason='harmful content violates policy',
        expected='block', risk_level=3,
        flags={'escalation_needed': False, 'encoded_detected': False,
               'emotional_manip': False, 'policy_conflict': False, 'roleplay_attempt': False},
        triggered_rules=['P001'], exceptions_apply=False, memory=m
    )
    s = result['step_score']
    assert 0.0 < s < 1.0, f'FAIL: step_score={s}'
    m.add_turn(turn, 'query', 'block', 'reason', 3, {})

final = g.final_score(m)
fs = final['final_score']
assert 0.0 < fs < 1.0, f'FAIL: final_score={fs}'
print(f'Grader: step OK, final={fs} OK')

# Zero-score path
g2 = Grader('easy', 3)
m2 = MemoryEngine()
z = g2.final_score(m2)
assert z['final_score'] > 0.0, f'FAIL: zero_score={z["final_score"]}'
print(f'Zero-score path: {z["final_score"]} OK')

print('ALL ASSERTIONS PASSED')