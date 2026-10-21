
from pathlib import Path
import pandas as pd

# Read raw transaction data
DATA_DIR = Path(__file__).resolve().parent
df = pd.read_csv(DATA_DIR / "transactions_raw.csv")

original_copy = df.copy(deep=True)
print("Original data saved for comparison.")

# Display basic information
print("Original shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:")
print(df.isna().sum())

audit_df = df.drop_duplicates()

print("\nAudit after removing duplicates:")
print("Rows:", len(audit_df))
print("Missing amount_inr:", audit_df["amount_inr"].isna().sum())
print("Missing distance_from_home_km:",
      audit_df["distance_from_home_km"].isna().sum())

out_of_range = (
    (audit_df["amount_inr"] < 1) |
    (audit_df["amount_inr"] > 1_000_000)
)
print("Out-of-range amounts:", out_of_range.sum())

print("\nDistinct raw spellings:")
print("merchant_category:", df["merchant_category"].nunique())
print("city:", df["city"].nunique())



def clean_transactions(raw_df):
    df = raw_df.copy()

    df = df.drop_duplicates()

    for col in ["merchant_category", "city"]:
        df[col] = df[col].str.strip().str.title()

    df = df.dropna(subset=["amount_inr"])

    df = df[
        (df["amount_inr"] >= 1) &
        (df["amount_inr"] <= 1_000_000)
    ]

    df = df.reset_index(drop=True)

    return df

def add_features(df):
    df = df.copy()

    df["amount_vs_avg_ratio"] = (
        df["amount_inr"] / df["avg_amount_30d_inr"]
    )

    df["is_night"] = (df["hour_of_day"] <= 5).astype(int)

    df["high_value"] = (df["amount_inr"] > 50000).astype(int)

    return df

cleaned_df = clean_transactions(df)
print("Original data unchanged:", df.equals(original_copy))
df = add_features(cleaned_df)

print("Shape after removing duplicates:", df.shape)
print("Remaining duplicate rows:", df.duplicated().sum())

print("\nCategory counts:")
print(df["merchant_category"].value_counts())

print("\nCity names:")
print(sorted(df["city"].unique()))

print("\nPayment modes:")
print(df["payment_mode"].value_counts())

print("\nMissing values:")
print(df.isna().sum())

print("\nNew feature columns:")
print(df[["amount_vs_avg_ratio", "is_night", "high_value"]].head())
print("\nFinal shape:", df.shape)

print("\nChargeback leakage audit:")
print(pd.crosstab(df["chargeback_filed"], df["is_fraud"]))

df.to_csv(DATA_DIR / "transactions_cleaned.csv", index=False)

# Test functions with a single row
one_row = clean_transactions(original_copy.head(1))
one_row = add_features(one_row)

print("\nOne-row test shape:", one_row.shape)
print(one_row[["amount_vs_avg_ratio", "is_night", "high_value"]])

print("Maximum amount_vs_avg_ratio:", df["amount_vs_avg_ratio"].max())

print("\nFraud audit:")
fraud_count = df["is_fraud"].sum()
total_count = len(df)

print("Fraud transactions:", fraud_count)
print("Fraud rate (%):", round(fraud_count / total_count * 100, 2))
print("Never-flag baseline accuracy (%):",
      round((df["is_fraud"] == 0).mean() * 100, 2))


print("\nAdditional acceptance checks:")
print("Minimum amount:", round(df["amount_inr"].min(), 2))
print("Maximum amount:", round(df["amount_inr"].max(), 2))

print("Night transactions:", df["is_night"].sum())
print("High-value transactions:", df["high_value"].sum())

# Never-flag baseline recall
baseline_recall = 0
print("Never-flag baseline recall (%):", baseline_recall)

# Chargeback row counts and fraud counts
print("\nChargeback summary:")
print(df.groupby("chargeback_filed")["is_fraud"].agg(
    total_rows="count",
    fraud_count="sum"
))
