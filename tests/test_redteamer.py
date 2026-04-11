# ============================================================
# SafetyGuard X — Red-Teamer Verification
# Standalone script to test Basilisk Mock generation.
# ============================================================

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.redteamer import redteamer

def test_mock_generation():
    print("--- Basilisk Red-Teamer: Mock Generation Test ---")
    
    contexts = [
        {
            "label": "Agent allowed a high-risk query",
            "ctx": {"last_decision": "allow", "risk_level": 4, "flags": {}}
        },
        {
            "label": "Agent detected roleplay but allowed",
            "ctx": {"last_decision": "allow", "risk_level": 2, "flags": {"roleplay_attempt": True}}
        },
        {
            "label": "Encoded attack detected",
            "ctx": {"last_decision": "block", "risk_level": 5, "flags": {"encoded_detected": True}}
        },
        {
            "label": "Standard fallback",
            "ctx": {"last_decision": "block", "risk_level": 1, "flags": {}}
        }
    ]

    for item in contexts:
        print(f"\nScenario: {item['label']}")
        for i in range(2):
            attack = redteamer.generate_attack(item['ctx'])
            print(f"  {i+1}. {attack}")
    
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_mock_generation()
