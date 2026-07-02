import re
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

print("Loading recommendation engine...")

df = pd.read_csv("data/shl_catalog_processed.csv")
index = faiss.read_index("retrieval/shl_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Recommendation engine ready!")


def detect_test_type(row):
    keys = str(row.get("keys", "")).lower()
    name = str(row.get("name", "")).lower()

    if "personality" in keys or "behavior" in keys or "opq" in name:
        return "P"

    if "ability" in keys or "aptitude" in keys or "verify" in name:
        return "A"

    return "K"


def search_assessments(query: str, top_k: int = 10):
    query_lower = query.lower()

    remote_filter = "remote" in query_lower
    adaptive_filter = "adaptive" in query_lower

    duration_limit = None
    match = re.search(r"(\d+)\s*minutes?", query_lower)

    if match:
        duration_limit = int(match.group(1))

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, 50)

    recommendations = []

    for idx in indices[0]:
        row = df.iloc[idx]

        if remote_filter and str(row["remote"]).lower() != "yes":
            continue

        if adaptive_filter and str(row["adaptive"]).lower() != "yes":
            continue

        if duration_limit:
            try:
                duration = int(re.findall(r"\d+", str(row["duration"]))[0])
                if duration > duration_limit:
                    continue
            except Exception:
                pass

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
        df["name"].str.lower().str.contains(name_query, na=False)
    ]

    if len(matches) == 0:
        matches = df[
            df["combined_text"].str.lower().str.contains(name_query, na=False)
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

    reply = (
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

    return reply