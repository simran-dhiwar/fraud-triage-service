
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 10

BENIGN = {
    "merchant_category": "Grocery",
    "city": "Pune",
    "payment_mode": "UPI",
    "amount_inr": 1500,
    "hour_of_day": 13,
    "txn_count_last_24h": 0,
    "avg_amount_30d_inr": 1600,
    "distance_from_home_km": 3.0,
    "is_new_device": 0,
    "account_age_days": 900,
}

RISKY = {
    "merchant_category": "Electronics",
    "city": "Mumbai",
    "payment_mode": "Card",
    "amount_inr": 45000,
    "hour_of_day": 2,
    "txn_count_last_24h": 9,
    "avg_amount_30d_inr": 3000,
    "distance_from_home_km": 60.0,
    "is_new_device": 1,
    "account_age_days": 40,
}

BIG = {
    **BENIGN,
    "merchant_category": "Utilities",
    "payment_mode": "NetBanking",
    "amount_inr": 150000,
    "avg_amount_30d_inr": 250000,
    "distance_from_home_km": 0.0,
    "is_new_device": 1,
    "account_age_days": 2200,
}


def make_case(txn_id, payload, question):
    return {
        **payload,
        "txn_id": txn_id,
        "question": question,
    }


tests = [
    {
        "id": "Q1",
        "category": "happy",
        "method": "GET",
        "path": "/health",
        "expected_status": 200,
        "checks": {"status": "ok"},
    },
    {
        "id": "Q2",
        "category": "happy",
        "method": "POST",
        "path": "/score",
        "payload": {**BENIGN, "txn_id": "T800001"},
        "expected_status": 200,
        "checks": {
            "flagged": False,
            "decision": "approve",
            "human_review": False,
        },
    },
    {
        "id": "Q3",
        "category": "escalation",
        "method": "POST",
        "path": "/score",
        "payload": {**RISKY, "txn_id": "T800002"},
        "expected_status": 200,
        "checks": {
            "flagged": True,
            "decision": "hold_for_review",
            "human_review": True,
        },
    },
    {
        "id": "Q4",
        "category": "escalation",
        "method": "POST",
        "path": "/score",
        "payload": {**BIG, "txn_id": "T800003"},
        "expected_status": 200,
        "checks": {
            "flagged": False,
            "decision": "hold_for_review",
            "human_review": True,
        },
    },
    {
        "id": "Q5",
        "category": "edge",
        "method": "POST",
        "path": "/score",
        "payload": {**BENIGN, "txn_id": "T800004", "amount_inr": -5},
        "expected_status": 400,
        "checks": {"amount_inr": True},
    },
    {
        "id": "Q6",
        "category": "edge",
        "method": "POST",
        "path": "/score",
        "payload": {
            key: value
            for key, value in {
                **BENIGN,
                "txn_id": "T800005",
            }.items()
            if key != "hour_of_day"
        },
        "expected_status": 422,
        "checks": {},
    },
    {
        "id": "Q7",
        "category": "escalation",
        "method": "POST",
        "path": "/case",
        "payload": make_case(
            "T900002",
            RISKY,
            "more than 6 UPI payments in 24 hours velocity alert",
        ),
        "expected_status": 200,
        "checks": {
            "flagged": True,
            "attempts": 2,
            "safe_default": False,
            "human_review": True,
            "case_note.policy_id": "P2",
        },
    },
    {
        "id": "Q8",
        "category": "escalation",
        "method": "POST",
        "path": "/case",
        "payload": make_case(
            "T900005",
            RISKY,
            "electronics merchant payment on an account younger than 30 days",
        ),
        "expected_status": 200,
        "checks": {
            "attempts": 3,
            "safe_default": True,
            "decision": "hold_for_review",
            "human_review": True,
        },
    },
    {
        "id": "Q9",
        "category": "edge",
        "method": "POST",
        "path": "/case",
        "payload": make_case(
            "T900099",
            RISKY,
            "card payment above INR 50000 from a new device",
        ),
        "expected_status": 503,
        "checks": {},
    },
    {
        "id": "Q10",
        "category": "happy",
        "method": "POST",
        "path": "/case",
        "payload": make_case(
            "T800001",
            BENIGN,
            "card payment above INR 50000 from a new device",
        ),
        "expected_status": 200,
        "checks": {
            "case_note": None,
            "attempts": 0,
            "decision": "approve",
            "human_review": False,
        },
    },
    {
        "id": "Q11",
        "category": "edge",
        "method": "POST",
        "path": "/case",
        "payload": make_case(
            "T800006",
            RISKY,
            "what is the cafeteria menu today",
        ),
        "expected_status": 200,
        "checks": {
            "case_note": None,
            "attempts": 0,
            "decision": "hold_for_review",
            "human_review": True,
        },
    },
]


def get_nested(data, key):
    for part in key.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data



def check_response(test, response):
    if response.status_code != test["expected_status"]:
        print(
            f'{test["id"]}: status mismatch — '
            f'expected {test["expected_status"]}, '
            f'got {response.status_code}'
        )
        return False

    try:
        body = response.json()
    except ValueError:
        print(f'{test["id"]}: response is not valid JSON')
        return False

    for key, expected in test["checks"].items():
        if key == "amount_inr":
            actual = "amount_inr" in str(body)
        else:
            actual = get_nested(body, key)

        if actual != expected:
            print(
                f'{test["id"]}: {key} mismatch — '
                f'expected {expected!r}, got {actual!r}'
            )
            print(f'Response body: {body}')
            return False

    return True



def wait_for_server():
    print("Waiting for the API server...")

    for _ in range(120):
        try:
            response = requests.get(
                f"{BASE_URL}/health",
                timeout=2,
            )
            if response.status_code == 200:
                print("Server is ready!\n")
                return True
        except requests.RequestException:
            pass

        time.sleep(1)

    print("Server did not become ready within 120 seconds.")
    return False


def main():
    if not wait_for_server():
        return

    results = []

    print("-" * 75)
    print(f"{'ID':<6}{'Category':<14}{'Expected':<12}{'Observed':<12}Result")
    print("-" * 75)

    for test in tests:
        try:
            if test["method"] == "GET":
                response = requests.get(
                    BASE_URL + test["path"],
                    timeout=TIMEOUT,
                )
            else:
                response = requests.post(
                    BASE_URL + test["path"],
                    json=test["payload"],
                    timeout=TIMEOUT,
                )

            passed = check_response(test, response)
            observed = response.status_code

        except requests.RequestException as exc:
            passed = False
            observed = "ERROR"
            print(f"{test['id']}: Request failed: {exc}")

        result = "PASS" if passed else "FAIL"
        results.append((test, passed))

        print(
            f"{test['id']:<6}"
            f"{test['category']:<14}"
            f"{test['expected_status']:<12}"
            f"{str(observed):<12}"
            f"{result}"
        )

    print("-" * 75)
    print(f"Total: {sum(p for _, p in results)}/11 passed")

    for category in ("happy", "escalation", "edge"):
        selected = [
            passed
            for test, passed in results
            if test["category"] == category
        ]
        print(
            f"{category}: {sum(selected)}/{len(selected)} passed"
        )

    try:
        response = requests.get(
            f"{BASE_URL}/metrics",
            timeout=TIMEOUT,
        )
        print("\nFinal /metrics:")
        print(response.text)
    except requests.RequestException as exc:
        print(f"Could not fetch metrics: {exc}")


if __name__ == "__main__":
    main()
