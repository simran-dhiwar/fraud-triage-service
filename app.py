
from pathlib import Path
import pandas as pd
import time
import logging

logging.basicConfig(
    filename="service.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

from casenote import run_case
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from prepare import clean_transactions, add_features
from train import build_final_model

app = FastAPI()
@app.middleware("http")
async def track_requests(request: Request, call_next):
    if request.method != "POST" or request.url.path not in {
        "/score", "/case"
    }:
        return await call_next(request)

    start = time.perf_counter()
    status_code = 500
    response = None
    txn_id = "unknown"
    probability = "unknown"

    try:
        body = await request.json()
        txn_id = body.get("txn_id", "unknown")
    except Exception:
        pass

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception("Unhandled request error")
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000

        METRICS["requests_total"] += 1
        METRICS["total_latency_ms"] += latency_ms

        if status_code < 400:
            METRICS["success_total"] += 1
        else:
            METRICS["error_total"] += 1

        level = (
            logging.ERROR if status_code >= 500
            else logging.WARNING if status_code >= 400
            else logging.INFO
        )

        logger.log(
            level,
            "endpoint=%s status=%s latency_ms=%.2f txn_id=%s probability=%s",
            request.url.path,
            status_code,
            latency_ms,
            txn_id,
            getattr(request.state, "probability", "unknown")
        )
        

METRICS = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "human_review_total": 0,
    "total_latency_ms": 0.0
}

@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    if exc.status_code == 400:
        return JSONResponse(
            status_code=400,
            content=exc.detail
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Train the model once when the service starts
MODEL = build_final_model()


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = None
    is_new_device: int
    account_age_days: int

class CaseIn(Txn):
    question: str
def score_transaction(txn: Txn):
    # Convert transaction into a DataFrame
    row = pd.DataFrame([txn.model_dump()])

    # Standardize text fields
    row["merchant_category"] = (
        row["merchant_category"].str.strip().str.title()
    )
    row["city"] = row["city"].str.strip().str.title()

    # Create the same features used during training
    row = add_features(row)

    # Remove fields that were not used for training
    X = row.drop(
        columns=[
            "txn_id",
            "customer_id",
            "chargeback_filed",
            "is_fraud"
        ],
        errors="ignore"
    )

    # Convert text categories into numeric columns
    X = pd.get_dummies(X, dtype=int)

    # Match the exact training columns and their order
    X = X.reindex(columns=MODEL["columns"], fill_value=0)

    # Fill missing distance using the training median
    X["distance_from_home_km"] = (
        X["distance_from_home_km"].fillna(MODEL["median"])
    )

    # Predict fraud probability
    probability = MODEL["model"].predict_proba(X)[0, 1]

    return round(float(probability), 4)

def validate_transaction(txn: Txn):
    data = txn.model_dump()

    if not 1 <= data["amount_inr"] <= 1_000_000:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "amount_inr must be between 1 and 1000000",
                "field": "amount_inr"
            }
        )

    if not 0 <= data["hour_of_day"] <= 23:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "hour_of_day must be between 0 and 23",
                "field": "hour_of_day"
            }
        )

    if data["txn_count_last_24h"] < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "txn_count_last_24h cannot be negative",
                "field": "txn_count_last_24h"
            }
        )

    if data["avg_amount_30d_inr"] <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "avg_amount_30d_inr must be greater than 0",
                "field": "avg_amount_30d_inr"
            }
        )

    allowed_categories = {
        "Grocery",
        "Food Delivery",
        "Fashion",
        "Electronics",
        "Utilities",
        "Travel"
    }

    category = data["merchant_category"].strip().title()

    if category not in allowed_categories:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown merchant category",
                "field": "merchant_category"
            }
        )

@app.post("/score")
def score(txn: Txn, request: Request):
    # Validate before running the model
    validate_transaction(txn)

    # Calculate fraud probability
    probability = score_transaction(txn)
    request.state.probability = probability

    # Decide whether to flag the transaction
    flagged = probability >= MODEL["threshold"]

    decision = "hold_for_review" if flagged else "approve"

    # Large payment from a new device must be reviewed
    if (
        not flagged
        and txn.amount_inr >= 100_000
        and txn.is_new_device == 1
    ):
        decision = "hold_for_review"

    human_review = decision == "hold_for_review"
    if human_review:
        METRICS["human_review_total"] += 1

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": probability,
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review
    }


@app.post("/case")
def create_case(case: CaseIn, request: Request):
    validate_transaction(case)

    probability = score_transaction(case)
    request.state.probability = probability

    flagged = probability >= MODEL["threshold"]
    if not flagged:
        result = {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False,
        }
    else:
        try:
            result = run_case(
                txn_id=case.txn_id,
                fraud_probability=probability,
                amount_inr=case.amount_inr,
                question=case.question
            )
        except ConnectionError:
            return JSONResponse(
                status_code=503,
                content={
                    "error": (
                        "case-note service temporarily unavailable, "
                        "try again later"
                    )
                }
            )
    case_note = result.get("case_note")

    decision = (
        case_note["recommended_action"]
        if case_note is not None
        else ("hold_for_review" if flagged else "approve")
    )

    # Large payment from a new device needs review.
    if (
        decision == "approve"
        and case.amount_inr >= 100_000
        and case.is_new_device == 1
    ):
        decision = "hold_for_review"

    # High fraud probability must never result in approval.
    guardrail_override = False
    if flagged and decision == "approve":
        decision = "hold_for_review"
        guardrail_override = True

    human_review = decision == "hold_for_review"

    if human_review:
        METRICS["human_review_total"] += 1

    return {
        "txn_id": case.txn_id,
        "fraud_probability": probability,
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review,
        **{
            key: value
            for key, value in result.items()
            if key not in {
                "txn_id",
                "human_review",
                "guardrail_override"
            }
        },
        "guardrail_override": (
            result.get("guardrail_override", False)
            or guardrail_override
        )
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    total = METRICS["requests_total"]

    return {
        "requests_total": total,
        "success_total": METRICS["success_total"],
        "error_total": METRICS["error_total"],
        "success_rate": round(
            METRICS["success_total"] / total, 4
        ) if total else 0.0,
        "error_rate": round(
            METRICS["error_total"] / total, 4
        ) if total else 0.0,
        "avg_latency_ms": round(
            METRICS["total_latency_ms"] / total, 2
        ) if total else 0.0,
        "human_review_total": METRICS["human_review_total"],
        "human_review_rate": round(
            METRICS["human_review_total"] / total, 4
        ) if total else 0.0
    }