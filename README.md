fairforge/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry — keep mostly same, update routes
│   ├── policies.py          # REPLACE FULLY — 12 fairness policies
│   ├── grader.py            # UPDATE — 6-metric fairness grader
│   ├── adversary.py         # UPDATE — bias injector instead of jailbreak
│   ├── fairness_metrics.py  # NEW — core fairness math
│   ├── mitigation_engine.py # NEW — fix suggestions
│   └── gemini_auditor.py    # NEW — Gemini API integration
├── data/
│   └── tasks/
│       ├── hiring_easy.json
│       ├── loan_medium.json
│       ├── medical_hard.json
│       └── intersectional_expert.json
├── openenv/                 # KEEP EXACTLY AS IS
│   ├── env.py
│   ├── ppo_trainer.py
│   └── basilisk.py
└── reports/                 # NEW — exported fairness reports