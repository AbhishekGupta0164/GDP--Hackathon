import sys
sys.path.insert(0, '.')
from app.grader import _clamp, _SCORE_MIN, _SCORE_MAX, Grader
from app.memory import MemoryEngine

# 1. Clamp boundaries
assert _clamp(0.0) == 0.001
assert _clamp(1.0) == 0.999
assert _clamp(0.5) == 0.5
assert 0.0 < _clamp(-999) < 1.0
assert 0.0 < _clamp(999)  < 1.0
print('clamp: OK')

# 2. All decisions x all tasks x all risk levels
flags_safe     = {'escalation_needed':False,'encoded_detected':False,'emotional_manip':False,'policy_conflict':False,'roleplay_attempt':False}
flags_critical = {'escalation_needed':True, 'encoded_detected':True, 'emotional_manip':True, 'policy_conflict':True,'roleplay_attempt':True}

for task, max_t in [('easy',3),('medium',5),('hard',7),('expert',10)]:
    for dec in ['allow','block','modify','escalate','clarify']:
        for flg in [flags_safe, flags_critical]:
            for risk in [0, 3, 5]:
                g = Grader(task, max_t)
                m = MemoryEngine()
                r = g.score_turn(1, dec, 'This violates policy P001 harmful dangerous content', 'block', risk, flg, ['P001'], False, m)
                s = r['step_score']
                assert 0.0 < s < 1.0, f'FAIL step: task={task} dec={dec} risk={risk} score={s}'
                m.add_turn(1, 'q', dec, 'r', risk, {})
                final = g.final_score(m)
                fs = final['final_score']
                assert 0.0 < fs < 1.0, f'FAIL final: task={task} dec={dec} risk={risk} score={fs}'
    print(f'{task}: all decisions x all risk-levels OK')

# 3. Zero-score path (no turns)
g2 = Grader('easy', 3)
m2 = MemoryEngine()
z = g2.final_score(m2)
assert z['final_score'] > 0.0, f'FAIL zero: {z["final_score"]}'
print(f'zero-score path: {z["final_score"]} OK')

# 4. Multi-turn episode
g3 = Grader('expert', 10)
m3 = MemoryEngine()
for turn in range(1, 6):
    r = g3.score_turn(turn, 'escalate', 'Critical encoded content detected violates policy P007 dangerous', 'escalate', 5, flags_critical, ['P007'], False, m3)
    s = r['step_score']
    assert 0.0 < s < 1.0, f'FAIL multi-turn step {turn}: {s}'
    m3.add_turn(turn, 'dangerous query', 'escalate', 'reason', 5, {})
final3 = g3.final_score(m3)
fs3 = final3['final_score']
assert 0.0 < fs3 < 1.0, f'FAIL multi-turn final: {fs3}'
print(f'multi-turn expert: final={fs3} OK')

print('ALL ASSERTIONS PASSED - SAFE TO PUSH')
