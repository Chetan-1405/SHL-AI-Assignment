from fastapi import FastAPI
from models.schemas import ChatRequest, ChatResponse
from retrieval.recommender import search_assessments, compare_assessments

app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="3.1.0"
)


@app.get("/")
def root():
    return {"message": "SHL Conversational Assessment Recommender"}


@app.get("/health")
def health():
    return {"status": "ok"}


def is_off_topic(text: str) -> bool:
    text = text.lower()

    off_topic_words = [
        "legal",
        "law",
        "salary",
        "resume",
        "interview tips",
        "hiring advice",
        "fire employee",
        "termination",
        "contract",
        "policy",
        "medical",
        "doctor",
        "weather",
        "movie",
        "sports",
    ]

    return any(word in text for word in off_topic_words)


def is_vague(text: str) -> bool:
    text = text.lower().strip()

    vague_queries = [
        "assessment",
        "i need an assessment",
        "need assessment",
        "suggest assessment",
        "recommend assessment",
        "recommend test",
        "suggest test",
        "test",
    ]

    if text in vague_queries:
        return True

    if len(text.split()) <= 3:
        return True

    return False


def is_comparison_query(text: str) -> bool:
    text = text.lower()
    return (
        "difference between" in text
        or "compare" in text
        or " vs " in text
        or " versus " in text
    )


def extract_comparison_names(text: str):
    lower = text.lower()

    known_names = {
        "opq": "OPQ",
        "opq32": "OPQ",
        "opq32r": "OPQ",
        "gsa": "Global Skills Assessment",
        "global skills assessment": "Global Skills Assessment",
        "java": "Java",
        "verify": "Verify",
    }

    found = []

    for key, value in known_names.items():
        if key in lower and value not in found:
            found.append(value)

    if len(found) >= 2:
        return found[0], found[1]

    return None, None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = request.messages

    if not messages:
        return {
            "reply": "Please tell me what role, skill, or assessment need you are hiring for.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    user_messages = [
        msg.content for msg in messages if msg.role == "user"
    ]

    latest_user_message = user_messages[-1]
    full_context = " ".join(user_messages)

    if is_off_topic(latest_user_message):
        return {
            "reply": "I can only help with SHL assessment recommendations, refinements, and comparisons based on the SHL catalog.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    if is_comparison_query(latest_user_message):
        name1, name2 = extract_comparison_names(latest_user_message)

        if name1 and name2:
            comparison_reply = compare_assessments(name1, name2)

            if comparison_reply:
                return {
                    "reply": comparison_reply,
                    "recommendations": [],
                    "end_of_conversation": False,
                }

        return {
            "reply": "I can compare SHL assessments from the catalog. Please mention two assessment names, for example OPQ and GSA.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    if is_vague(full_context):
        return {
            "reply": "Sure. To recommend the right SHL assessments, please tell me the role, skills to assess, seniority level, and any constraints such as duration, remote testing, or adaptive testing.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    recommendations = search_assessments(full_context, top_k=5)

    formatted_recommendations = []

    for item in recommendations:
        formatted_recommendations.append(
            {
                "name": item["assessment_name"],
                "url": item["url"],
                "test_type": item.get("test_type"),
            }
        )

    return {
        "reply": f"Here are {len(formatted_recommendations)} SHL assessments that best match your requirement.",
        "recommendations": formatted_recommendations,
        "end_of_conversation": True,
    }