import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading lightweight recommendation engine...")

df = pd.read_csv("data/shl_catalog_processed.csv")

df["combined_text"] = df["combined_text"].fillna("")
df["name"] = df["name"].fillna("")
df["description"] = df["description"].fillna("")
df["keys"] = df["keys"].fillna("")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=8000
)

tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

print("Lightweight recommendation engine ready!")


def detect_test_type(row):
    keys = str(row.get("keys", "")).lower()
    name = str(row.get("name", "")).lower()

    if "personality" in keys or "behavior" in keys or "opq" in name:
        return "P"

    if "ability" in keys or "aptitude" in keys or "verify" in name:
        return "A"

    return "K"


def extract_duration_minutes(value):
    nums = re.findall(r"\d+", str(value))
    if nums:
        return int(nums[0])
    return None


def search_assessments(query: str, top_k: int = 10):
    query_lower = query.lower()

    remote_filter = "remote" in query_lower
    adaptive_filter = "adaptive" in query_lower

    duration_limit = None
    match = re.search(
        r"(under|below|less than|within)?\s*(\d+)\s*minutes?",
        query_lower
    )

    if match:
        duration_limit = int(match.group(2))

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_indices = similarities.argsort()[::-1]

    recommendations = []

    for idx in ranked_indices:
        row = df.iloc[idx]

        if similarities[idx] <= 0:
            continue

        if remote_filter and str(row["remote"]).lower() != "yes":
            continue

        if adaptive_filter and str(row["adaptive"]).lower() != "yes":
            continue

        duration = extract_duration_minutes(row["duration"])

        if duration_limit and duration:
            if duration > duration_limit:
                continue

        recommendations.append(
            {
                "assessment_name": row["name"],
                "url": row["link"],
                "duration": str(row["duration"]),
                "remote_testing": str(row["remote"]),
                "adaptive": str(row["adaptive"]),
                "description": str(row["description"]),
                "test_type": detect_test_type(row),
            }
        )

        if len(recommendations) >= top_k:
            break

    return recommendations


def find_assessment_by_name(name_query: str):
    name_query = name_query.lower().strip()

    matches = df[
        df["name"].str.lower().str.contains(name_query, na=False, regex=False)
    ]

    if len(matches) == 0:
        matches = df[
            df["combined_text"].str.lower().str.contains(
                name_query,
                na=False,
                regex=False
            )
        ]

    if len(matches) == 0:
        return None

    row = matches.iloc[0]

    return {
        "name": row["name"],
        "url": row["link"],
        "duration": str(row["duration"]),
        "remote": str(row["remote"]),
        "adaptive": str(row["adaptive"]),
        "description": str(row["description"]),
        "test_type": detect_test_type(row),
        "keys": str(row.get("keys", "")),
    }


def compare_assessments(name1: str, name2: str):
    a = find_assessment_by_name(name1)
    b = find_assessment_by_name(name2)

    if not a or not b:
        return None

    return (
        f"{a['name']} and {b['name']} are different SHL assessments.\n\n"
        f"{a['name']}:\n"
        f"- Test type: {a['test_type']}\n"
        f"- Duration: {a['duration']}\n"
        f"- Remote testing: {a['remote']}\n"
        f"- Adaptive: {a['adaptive']}\n"
        f"- URL: {a['url']}\n"
        f"- Description: {a['description']}\n\n"
        f"{b['name']}:\n"
        f"- Test type: {b['test_type']}\n"
        f"- Duration: {b['duration']}\n"
        f"- Remote testing: {b['remote']}\n"
        f"- Adaptive: {b['adaptive']}\n"
        f"- URL: {b['url']}\n"
        f"- Description: {b['description']}"
    )