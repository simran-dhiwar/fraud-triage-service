import json

from pathlib import Path

from mock_llm import generate, reset
from retrieval import retrieve, MIN_SCORE
def build_prompt(txn_id, fraud_probability, amount_inr, question, policy):
    policy_id = policy[0]

    return (
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {fraud_probability}\n"
        f"AMOUNT_INR: {amount_inr}\n"
        f"POLICY_ID: {policy_id}\n"
        f"QUESTION: {question}\n"
        f"POLICY: {json.dumps(policy)}\n"
    )

def validate(text, retrieved_policy_id):
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, 1

    if not isinstance(note, dict):
        return False, 1

    required_keys = [
        "summary",
        "priority",
        "recommended_team",
        "recommended_action",
        "policy_id"
    ]

    if any(key not in note for key in required_keys):
        return False, 2

    if any(not isinstance(note[key], str) for key in required_keys):
        return False, 2

    if note["priority"] not in {"low", "medium", "high"}:
        return False, 3

    if note["recommended_team"] not in {
        "card_risk",
        "upi_risk",
        "merchant_ops",
        "customer_care"
    }:
        return False, 3

    if note["recommended_action"] not in {
        "approve",
        "hold_for_review",
        "block_and_notify"
    }:
        return False, 3

    if note["policy_id"] not in {"P1", "P2", "P3", "P4", "P5", "P6"}:
        return False, 3

    if note["policy_id"] != retrieved_policy_id:
        return False, 3

    return True, 0

def run_case(
    txn_id,
    fraud_probability,
    amount_inr,
    question,
    max_attempts=3
):
    policy_id, policy_text, score = retrieve(question)

    if score < MIN_SCORE:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False
        }

    policy = (policy_id, policy_text, score)

    prompt = build_prompt(
        txn_id,
        fraud_probability,
        amount_inr,
        question,
        policy
    )

    failures = []
    case_note = None
    safe_default = False

    for attempt in range(1, max_attempts + 1):
        response = generate(prompt)

        valid, gate = validate(response, policy_id)

        if valid:
            case_note = json.loads(response)
            break

        failures.append(gate)

    if case_note is None:
        safe_default = True
        case_note = {
            "summary": (
                "Automated case note unavailable; "
                "manual review required."
            ),
            "priority": "high",
            "recommended_team": "customer_care",
            "recommended_action": "hold_for_review",
            "policy_id": policy_id
        }

    guardrail_override = False

    if (
        case_note["recommended_action"] == "approve"
        and fraud_probability >= 0.5
    ):
        case_note["recommended_action"] = "hold_for_review"
        guardrail_override = True

    return {
        "status": "ok",
        "attempts": min(len(failures) + 1, max_attempts),
        "failures": failures,
        "case_note": case_note,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override
    }

if __name__ == "__main__":
    reset()

    with open(
        Path(__file__).resolve().parent / "case_scenarios.json",
        "r",
        encoding="utf-8"
    ) as file:
        scenarios = json.load(file)

    print("\nCASE NOTE TEST RESULTS")
    print("-" * 80)

    for scenario in scenarios:
        result = run_case(
            txn_id=scenario["txn_id"],
            fraud_probability=scenario["fraud_probability"],
            amount_inr=scenario["amount_inr"],
            question=scenario["question"]
        )

        print(
            f"{scenario['txn_id']} | "
            f"Status: {result['status']} | "
            f"Attempts: {result['attempts']} | "
            f"Failures: {result['failures']} | "
            f"Safe default: {result['safe_default']} | "
            f"Override: {result['guardrail_override']}"
        )