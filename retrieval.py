
from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).resolve().parent

# Load the six fraud policies
with open(DATA_DIR / "fraud_policies.json", "r", encoding="utf-8") as file:
    policies = json.load(file)

# Learn TF-IDF from policy text only
vectorizer = TfidfVectorizer()
policy_texts = [policy["text"] for policy in policies]
policy_matrix = vectorizer.fit_transform(policy_texts)


def retrieve(query):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, policy_matrix)[0]

    best_index = scores.argmax()
    best_policy = policies[best_index]

    return (
        best_policy["policy_id"],
        best_policy["text"],
        float(scores[best_index])
    )


# Measure scores using the supplied queries
with open(DATA_DIR / "retrieval_queries.json", "r", encoding="utf-8") as file:
    queries = json.load(file)

results = []

for item in queries:
    policy_id, policy_text, score = retrieve(item["query"])

    results.append({
        "query": item["query"],
        "expected": item["expected_policy"],
        "retrieved": policy_id,
        "score": score
    })

in_scope_scores = [
    result["score"] for result in results
    if result["expected"] is not None
]

out_of_scope_scores = [
    result["score"] for result in results
    if result["expected"] is None
]

# Choose a refusal threshold between the two score groups
lowest_in_scope = min(in_scope_scores)
highest_out_of_scope = max(out_of_scope_scores)

if highest_out_of_scope < lowest_in_scope:
    MIN_SCORE = (highest_out_of_scope + lowest_in_scope) / 2
else:
    raise ValueError(
        "The in-scope and out-of-scope scores overlap. "
        "A valid MIN_SCORE cannot be chosen automatically."
    )


if __name__ == "__main__":
    print("RETRIEVAL RESULTS")
    print("-" * 100)

    for result in results:
        print(
            f"Expected: {str(result['expected']):5} | "
            f"Retrieved: {result['retrieved']} | "
            f"Score: {result['score']:.4f} | "
            f"Query: {result['query']}"
        )

    hits = sum(
        result["expected"] == result["retrieved"]
        for result in results
        if result["expected"] is not None
    )

    print(f"\nIn-scope top-1 hits: {hits}/8")
    print(f"Highest out-of-scope score: {highest_out_of_scope:.4f}")
    print(f"Lowest in-scope score: {lowest_in_scope:.4f}")
    print(f"Chosen MIN_SCORE: {MIN_SCORE:.4f}")
