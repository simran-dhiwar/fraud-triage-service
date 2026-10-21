
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from prepare import clean_transactions, add_features

# Project folder
DATA_DIR = Path(__file__).resolve().parent

# Load and prepare data
raw = pd.read_csv(DATA_DIR / "transactions_raw.csv")
df = add_features(clean_transactions(raw))

# Separate target and input features
y = df["is_fraud"]

excluded_columns = [
    "txn_id",
    "customer_id",
    "chargeback_filed",
    "is_fraud"
]

X = df.drop(columns=excluded_columns)

# Convert categories into numeric columns
X = pd.get_dummies(X, dtype=int)

# Keep the exact input column order
columns = list(X.columns)

print("\nModel input columns:")
print(columns)
print("Total input columns:", len(columns))

# First split: 60% training, 40% remaining
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

# Second split: remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_remaining, y_remaining,
    test_size=0.5,
    stratify=y_remaining,
    random_state=42
)

# Fill missing distances using the training-set median only
median_distance = X_train["distance_from_home_km"].median()

for split in [X_train, X_val, X_test]:
    split["distance_from_home_km"] = (
        split["distance_from_home_km"].fillna(median_distance)
    )

# Display split results
print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nFraud counts:")
print("Training:", y_train.sum())
print("Validation:", y_val.sum())
print("Test:", y_test.sum())

print("\nTraining distance median:", median_distance)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Create three Logistic Regression models
models = {
    "Plain": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Balanced": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),
    "Oversampled": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
}

# Oversample fraud rows from the training data only
train_data = X_train.copy()
train_data["is_fraud"] = y_train

fraud_rows = train_data[train_data["is_fraud"] == 1]
normal_rows = train_data[train_data["is_fraud"] == 0]

fraud_extra = fraud_rows.sample(
    n=len(normal_rows),
    replace=True,
    random_state=42
)

oversampled_data = pd.concat(
    [normal_rows, fraud_extra]
).sample(frac=1, random_state=42)

X_over = oversampled_data.drop(columns=["is_fraud"])
y_over = oversampled_data["is_fraud"]

print("\nOversampled training rows:", len(X_over))

# Train and evaluate the three models
for name, model in models.items():
    if name == "Oversampled":
        model.fit(X_over, y_over)
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print(f"\n{name} Logistic Regression")
    print("Precision:", round(
        precision_score(y_val, predictions, zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_val, predictions, zero_division=0), 3
    ))
    print("F1:", round(
        f1_score(y_val, predictions, zero_division=0), 3
    ))
    print("Confusion matrix (TN, FP, FN, TP):",
          confusion_matrix(y_val, predictions).ravel().tolist())


from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Balanced Logistic Regression
balanced_model = models["Balanced"]

lr_scores = cross_val_score(
    balanced_model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

print("\nCross-validation results")
print("Balanced Logistic Regression mean F1:",
      round(lr_scores.mean(), 3))
print("Balanced Logistic Regression std F1:",
      round(lr_scores.std(), 3))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

rf_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

print("Random Forest mean F1:", round(rf_scores.mean(), 3))
print("Random Forest std F1:", round(rf_scores.std(), 3))

# Find the best Random Forest settings
param_grid = {
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [5, 10, 20]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nRandom Forest tuning results")
print("Best parameters:", grid_search.best_params_)
print("Best mean CV F1:", round(grid_search.best_score_, 3))


# Select the best Random Forest from tuning
best_rf = grid_search.best_estimator_

# Get fraud probabilities on validation data
val_probabilities = best_rf.predict_proba(X_val)[:, 1]

from sklearn.metrics import precision_score, recall_score, f1_score

threshold_results = []

# Try thresholds from 0.1 to 0.9
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": round(
            precision_score(y_val, val_predictions, zero_division=0), 3
        ),
        "recall": round(
            recall_score(y_val, val_predictions, zero_division=0), 3
        ),
        "f1": round(
            f1_score(y_val, val_predictions, zero_division=0), 3
        ),
        "flagged": int(val_predictions.sum())
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nValidation threshold results")
print(threshold_df.to_string(index=False))

# Choose the highest threshold with recall >= 0.65
eligible = threshold_df[threshold_df["recall"] >= 0.65]

if not eligible.empty:
    t_star = eligible["threshold"].max()
else:
    t_star = 0.1

print("\nSelected threshold:", t_star)


from sklearn.metrics import confusion_matrix

# Final evaluation on the untouched test set
test_probabilities = best_rf.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= t_star).astype(int)

test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)
test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)
test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

print("\nFINAL TEST RESULTS")
print("Threshold:", t_star)
print("Precision:", round(test_precision, 3))
print("Recall:", round(test_recall, 3))
print("F1:", round(test_f1, 3))
print("Confusion matrix (TN, FP, FN, TP):",
      [int(tn), int(fp), int(fn), int(tp)])
print("Total transactions flagged:", int(test_predictions.sum()))

if tp > 0:
    print("Legitimate transactions held per fraud caught:",
          round(fp / tp, 2))
else:
    print("Legitimate transactions held per fraud caught: N/A")


# Leakage experiment: add chargeback_filed as the 27th input
X_leaky = X.copy()
X_leaky["chargeback_filed"] = df["chargeback_filed"].values

# Use the same train/validation/test row positions
X_leaky_train = X_leaky.loc[X_train.index].copy()
X_leaky_val = X_leaky.loc[X_val.index].copy()

# Use the same training-only median imputation
for split in [X_leaky_train, X_leaky_val]:
    split["distance_from_home_km"] = (
        split["distance_from_home_km"].fillna(median_distance)
    )

# Train Random Forest with the previously selected parameters
leaky_rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    **grid_search.best_params_
)

leaky_rf.fit(X_leaky_train, y_train)

leaky_val_predictions = leaky_rf.predict(X_leaky_val)

leaky_f1 = f1_score(
    y_val, leaky_val_predictions, zero_division=0
)

clean_val_predictions = (
    best_rf.predict(X_val)
)

clean_f1 = f1_score(
    y_val, clean_val_predictions, zero_division=0
)

print("\nLeakage experiment")
print("Clean tuned Random Forest validation F1:",
      round(clean_f1, 3))
print("Leaky Random Forest validation F1 at 0.5:",
      round(leaky_f1, 3))


# Top 5 feature importances from the clean tuned model
importances = pd.Series(
    best_rf.feature_importances_,
    index=columns
).sort_values(ascending=False)

print("\nTop 5 Feature Importances")
print(importances.head(5).round(4).to_string())


def build_final_model():
    # Load and prepare data
    raw_data = pd.read_csv(DATA_DIR / "transactions_raw.csv")
    prepared = add_features(clean_transactions(raw_data))

    y_all = prepared["is_fraud"]

    excluded = [
        "txn_id",
        "customer_id",
        "chargeback_filed",
        "is_fraud"
    ]

    X_all = prepared.drop(columns=excluded)
    X_all = pd.get_dummies(X_all, dtype=int)

    # Save the exact 26-column order
    final_columns = list(X_all.columns)

    # Reproduce the required stratified splits
    X_tr, X_remaining, y_tr, y_remaining = train_test_split(
        X_all,
        y_all,
        test_size=0.4,
        stratify=y_all,
        random_state=42
    )

    X_va, X_te, y_va, y_te = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.5,
        stratify=y_remaining,
        random_state=42
    )

    # Impute using training data only
    training_median = X_tr["distance_from_home_km"].median()

    for split in [X_tr, X_va, X_te]:
        split["distance_from_home_km"] = (
            split["distance_from_home_km"].fillna(training_median)
        )

    # Tune Random Forest using training data only
    cv_splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ),
        param_grid={
            "max_depth": [4, 6, 8, None],
            "min_samples_leaf": [5, 10, 20]
        },
        scoring="f1",
        cv=cv_splitter,
        n_jobs=-1
    )

    search.fit(X_tr, y_tr)
    final_model = search.best_estimator_

    # Select threshold using validation recall >= 0.65
    probabilities = final_model.predict_proba(X_va)[:, 1]

    eligible_thresholds = []

    for threshold in [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9
    ]:
        predictions = (probabilities >= threshold).astype(int)

        recall = recall_score(
            y_va,
            predictions,
            zero_division=0
        )

        if recall >= 0.65:
            eligible_thresholds.append(threshold)

    final_threshold = (
        max(eligible_thresholds)
        if eligible_thresholds
        else 0.1
    )

    return {
        "model": final_model,
        "columns": final_columns,
        "median": training_median,
        "threshold": final_threshold
    }


print("\nTesting build_final_model()...")

final_result = build_final_model()

print("Returned keys:", list(final_result.keys()))
print("Number of columns:", len(final_result["columns"]))
print("Training median:", final_result["median"])
print("Selected threshold:", final_result["threshold"])
print("Model type:", type(final_result["model"]).__name__)
