# SHL AI Assessment Recommendation Agent

## Overview

This project is a conversational AI agent built for the SHL AI Research Intern assignment.

It recommends SHL Individual Test Solutions based on user requirements using the official SHL Product Catalog.

The application is developed using FastAPI and provides REST APIs for recommendation, clarification, refinement, and assessment comparison.

---

## Features

- Conversational recommendation system
- Clarifies vague user queries
- Recommends between 1–10 SHL assessments
- Supports conversation refinement
- Compares SHL assessments
- Rejects out-of-scope queries
- Uses the official SHL Product Catalog
- Public REST API

---

## Technologies Used

- Python
- FastAPI
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Uvicorn

---

## Folder Structure

```text
SHL-AI-Assignment/
│
├── api/
│   └── main.py
├── models/
│   └── schemas.py
├── retrieval/
│   └── recommender.py
├── preprocessing/
├── data/
├── embeddings/
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Go into the folder

```bash
cd SHL-AI-Assignment
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the API

```bash
uvicorn api.main:app --reload
```

---

## API Endpoints

### Health Check

GET

```
/health
```

Response

```json
{
    "status":"ok"
}
```

---

### Chat

POST

```
/chat
```

Example Request

```json
{
  "messages":[
    {
      "role":"user",
      "content":"I need a Java assessment under 30 minutes."
    }
  ]
}
```

---

## Deployment

Render URL

```
https://shl-ai-assignment-29ba.onrender.com
```

Swagger UI

```
https://shl-ai-assignment-29ba.onrender.com/docs
```

---

## Future Improvements

- Hybrid semantic retrieval
- Better ranking
- Personalized recommendations
- User feedback integration

---

## Author

Chetan Ventrapragada