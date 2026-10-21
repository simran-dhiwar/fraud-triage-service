
# Razorpay Merchant Risk Desk — Fraud Triage Service

## Project Overview

This project builds a fraud triage service for payment transactions.
It cleans transaction data, creates useful features, detects potentially
fraudulent transactions, retrieves relevant fraud policies, and generates
case notes.

## Part A: Data Foundation

### Raw Data Audit

- Original rows: 6,090
- Original columns: 14
- Exact duplicate rows: 90
- Missing amount values before deduplication: 79
- Missing distance values before deduplication: 196

After removing exact duplicates:

- Rows: 6,000
- Missing amount values: 76
- Missing distance values: 193
- Out-of-range amounts: 5
- Distinct raw merchant category spellings: 24
- Distinct raw city spellings: 24

### Data Cleaning

The `clean_transactions(raw_df)` function:

1. Creates a copy of the input DataFrame.
2. Removes duplicate rows.
3. Strips extra spaces and standardizes `merchant_category` and `city`.
4. Removes rows with missing `amount_inr`.
5. Removes amounts below ₹1 or above ₹1,000,000.
6. Resets the index.

The function does not modify the original DataFrame.

Missing values in `distance_from_home_km` are intentionally preserved
during Part A. Any imputation statistic must be learned using training
rows only, to avoid data leakage.

### Feature Engineering

The `add_features(df)` function creates:

- `amount_vs_avg_ratio`: transaction amount divided by the customer's
  average transaction amount over 30 days.
- `is_night`: 1 when `hour_of_day` is at most 5; otherwise 0.
- `high_value`: 1 when `amount_inr` is greater than ₹50,000; otherwise 0.

### Final Cleaned Dataset

- Rows: 5,919
- Columns: 17
- Duplicate rows: 0
- Missing `amount_inr`: 0
- Missing `distance_from_home_km`: 191
- Fraud transactions: 366
- Fraud rate: 6.18%

### Chargeback Leakage Audit

| chargeback_filed | Total rows | Fraud rows |
|---|---:|---:|
| 0 | 5,546 | 62 |
| 1 | 373 | 304 |

`chargeback_filed` must not be used as a model input because it may be
recorded after a transaction has been investigated or disputed. Using it
could reveal information that would not be available when the model
needs to make its decision.

`txn_id` is an identifier, not a behavioral signal. `customer_id` is
also excluded to prevent the model from relying on customer identity
rather than transaction characteristics.

The intended model inputs before one-hot encoding are:

- `merchant_category`
- `city`
- `payment_mode`
- `amount_inr`
- `hour_of_day`
- `txn_count_last_24h`
- `avg_amount_30d_inr`
- `distance_from_home_km`
- `is_new_device`
- `account_age_days`
- `amount_vs_avg_ratio`
- `is_night`
- `high_value`

The target column is `is_fraud`. It must never be included among the
model input features.

### Why Accuracy Can Be Misleading

Only 366 of 5,919 transactions are fraudulent, giving a fraud rate of
6.18%.

A baseline model that predicts every transaction as non-fraudulent
achieves 93.82% accuracy but detects zero fraudulent transactions.
Its fraud recall is 0%.

Therefore, accuracy alone is not a suitable headline metric for this
imbalanced fraud-detection problem. Recall and precision should also
be evaluated.

## Run the Data Preparation Script

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python prepare.py
```

## Part B — Fraud Model Training and Evaluation

### Data Split

- Training rows: 3,551
- Validation rows: 1,184
- Test rows: 1,184
- Fraud cases: 220 training, 73 validation, 73 test
- Missing `distance_from_home_km` values were filled using the training-set median of 12.6 km.
- Model input features: 26, excluding identifiers, `chargeback_filed`, and the target `is_fraud`.

### Logistic Regression Comparison

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Plain Logistic Regression | 0.625 | 0.342 | 0.442 |
| Balanced Logistic Regression | 0.252 | 0.781 | 0.381 |
| Oversampled Logistic Regression | 0.257 | 0.781 | 0.386 |

Oversampling was performed on the training data only, producing 6,662 rows.

### Cross-Validation and Tuning

- Balanced Logistic Regression mean CV F1: 0.402; standard deviation: 0.015.
- Random Forest mean CV F1: 0.453; standard deviation: 0.052.
- Best tuned Random Forest parameters: `max_depth=None`, `min_samples_leaf=5`.
- Best mean CV F1: 0.465.
- Five-fold stratified cross-validation and grid search were performed on the training split only.

### Threshold Selection

The threshold was selected using validation data. The highest tested threshold with recall of at least 0.65 was **0.4**. At this threshold, validation precision was 0.272, recall was 0.712, F1 was 0.394, and 191 transactions were flagged.

### Final Test Results

The tuned Random Forest was evaluated once on the untouched test set using threshold 0.4.

- Precision: 0.281
- Recall: 0.781
- F1-score: 0.413
- Confusion matrix (TN, FP, FN, TP): `[965, 146, 16, 57]`
- Transactions flagged: 203
- Legitimate transactions flagged per fraud caught: 2.56

### Risk Committee Note

The selected threshold of 0.4 prioritizes fraud detection, achieving 78.1% recall on the test set. It catches 57 of 73 fraud cases but misses 16, while incorrectly flagging 146 legitimate transactions. This means approximately 2.56 legitimate transactions require review for every fraud caught. A stricter threshold of 0.6 on validation data improves precision to 0.507 and F1 to 0.507, but reduces recall to 0.507 and flags only 73 transactions, illustrating the trade-off between review workload and missed fraud. The selected threshold should therefore be reviewed against the risk desk's operational capacity and the cost of missed fraud.

### Data Leakage

`chargeback_filed` is excluded from the production model because it may be recorded after a transaction is investigated or identified as fraudulent. Including it can leak information that would not be available when the original transaction needs to be scored.



## Part C — Policy Retrieval and Case-Note Generation

### Policy Retrieval

The `retrieval.py` module retrieves the most relevant fraud policy for a
given query.

- Fraud policies are loaded from `fraud_policies.json`.
- Text is converted into TF-IDF vectors.
- Cosine similarity is used to compare the query with the available policies.
- The policy with the highest similarity score is returned along with its
  policy ID and similarity score.

The retrieval evaluation uses the supplied queries in
`retrieval_queries.json`.

### Case-Note Generation and Validation

The `casenote.py` module generates structured case notes using the supplied
mock LLM implementation.

- Retrieved policy information and transaction details are used to prepare
  case notes.
- Generated responses are validated before being accepted.
- Invalid responses can be retried.
- If validation repeatedly fails, a safe default is used.
- Connection failures are handled so that temporary service errors do not
  silently produce an accepted case note.

The supplied `mock_llm.py` implementation is used for testing without
requiring a paid external LLM service.

### Case-Note Test Scenarios

The case-note tests cover successful generation, retries, validation
failures, safe defaults, and connection errors.

The `case_scenarios.json` file contains the supplied test scenarios.


## Part D — Fraud Triage API Service

### API Implementation

The `app.py` module implements a FastAPI service for transaction fraud
triage.

The service provides the following endpoints:

- `GET /health`: Checks whether the service is running.
- `POST /score`: Returns the transaction's fraud probability, flag status,
  decision, and human-review status.
- `POST /case`: Returns the scoring result along with case-note and
  guardrail information when applicable.
- `GET /metrics`: Reports request counts, error and success rates,
  average latency, and human-review statistics.

### Error Handling and Guardrails

- Invalid transaction values are handled through request validation.
- Invalid requests and server-side failures are handled with appropriate
  HTTP status codes.
- Case-note generation is skipped for transactions that are not flagged.
- Flagged transactions use policy retrieval and case-note validation.
- Large payments from recently observed devices can be held for review
  according to the configured guardrail.
- The service records request activity in `service.log`.

### Smoke-Test Results

The `smoke_test.py` script checks the API endpoints and expected responses.

| Test ID | Category | Expected Status | Observed Status | Result |
|---|---|---:|---:|---|
| Q1 | Happy path | 200 | 200 | PASS |
| Q2 | Happy path | 200 | 200 | PASS |
| Q3 | Escalation | 200 | 200 | PASS |
| Q4 | Escalation | 200 | 200 | PASS |
| Q5 | Edge case | 400 | 400 | PASS |
| Q6 | Edge case | 422 | 422 | PASS |
| Q7 | Escalation | 200 | 200 | PASS |
| Q8 | Escalation | 200 | 200 | PASS |
| Q9 | Edge case | 503 | 503 | PASS |
| Q10 | Happy path | 200 | 200 | PASS |
| Q11 | Edge case | 200 | 200 | PASS |

**Total: 11/11 tests passed.**

### Final Metrics

The `/metrics` endpoint returned the following values after the final smoke-test run:

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 138.35,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

### Service Logs

The `service.log` file records API request activity, including request
methods, paths, response status codes, latency, and transaction details.

Example log entries:

```text
2026-10-21 17:42:13,394 | INFO | endpoint=/score status=200 latency_ms=290.54 txn_id=T800001 probability=0.043
2026-10-21 17:42:13,896 | WARNING | endpoint=/score status=400 latency_ms=6.54 txn_id=T800004 probability=unknown
2026-10-21 17:42:14,497 | ERROR | endpoint=/case status=503 latency_ms=69.84 txn_id=T900099 probability=0.9597
```

These entries illustrate a successful request, a client-side validation error, and a server-side failure.

### Limitations

- The model is trained on the supplied dataset; real-world performance
  may differ.
- Fraud predictions are risk estimates and are not proof of fraud.
- The configured threshold involves a trade-off between detecting fraud
  and sending legitimate transactions for review.
- TF-IDF retrieval depends on textual similarity and may not always select
  the best policy.
- The supplied mock LLM is intended for testing and does not represent
  a production LLM integration.
- A production deployment would require further security, monitoring,
  reliability, and performance testing.
- Human review remains important for uncertain or high-risk transactions.
```
