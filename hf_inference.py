# hf_inference.py
# Calls HuggingFace Inference API instead of loading models locally
# Enables cloud deployment without RAM constraints
# ============================================================

import requests
import os
import time

# Get token from environment variable
# Never hardcode your token in code
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"}

# Model endpoints
EMOTION_URL       = "https://router.huggingface.co/hf-inference/models/j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_URL     = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
TOXICITY_URL      = "https://router.huggingface.co/hf-inference/models/unitary/toxic-bert"
NLI_URL           = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
SIMILARITY_URL    = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

def _call_api(url, payload, retries=5):
    """
    Calls HuggingFace Inference API with retry logic.
    Models sometimes need to warm up — retries handle this.
    """
    for attempt in range(retries):
        response = requests.post(url, headers=HEADERS, json=payload)
        result   = response.json()

        # Model is loading — wait and retry
        if isinstance(result, dict) and "estimated_time" in result:
            wait_time = result.get("estimated_time", 20)
            print(f"Model loading, waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time)
            continue

        # Error — retry
        if isinstance(result, dict) and "error" in result:
            print(f"API error: {result['error']} — retrying...")
            time.sleep(20)
            continue

        return result

    raise Exception(f"API call failed after {retries} retries")


def detect_emotion_api(text):
    """Detects emotion from text using HuggingFace API."""
    result = _call_api(EMOTION_URL, {"inputs": text})

    # Result is list of lists — flatten
    if isinstance(result[0], list):
        result = result[0]

    top = max(result, key=lambda x: x["score"])
    return {
        "emotion_label":    top["label"].lower(),
        "confidence_score": round(top["score"], 4)
    }


def analyze_sentiment_api(text):
    """Analyzes sentiment from text using HuggingFace API."""
    result = _call_api(SENTIMENT_URL, {"inputs": text})

    if isinstance(result[0], list):
        result = result[0]

    scores = {r["label"].lower(): r["score"] for r in result}

    positive = scores.get("positive", 0.0)
    negative = scores.get("negative", 0.0)
    sentiment_score = round(positive - negative, 4)

    if sentiment_score <= -0.6:
        category = "Strong Negative"
    elif sentiment_score <= -0.2:
        category = "Mild Negative"
    elif sentiment_score < 0.2:
        category = "Neutral"
    elif sentiment_score <= 0.6:
        category = "Mild Positive"
    else:
        category = "Strong Positive"

    return {
        "sentiment_score":    sentiment_score,
        "sentiment_category": category
    }


def analyze_toxicity_api(text):
    """Analyzes toxicity from text using HuggingFace API."""
    result = _call_api(TOXICITY_URL, {"inputs": text})

    if isinstance(result[0], list):
        result = result[0]

    scores = {r["label"].lower(): r["score"] for r in result}
    toxicity_score = round(scores.get("toxic", 0.0), 4)

    if toxicity_score <= 0.30:
        category = "Low"
    elif toxicity_score <= 0.50:
        category = "Moderate"
    else:
        category = "High"

    return {
        "toxicity_score":    toxicity_score,
        "toxicity_category": category
    }


def classify_zero_shot_api(text, candidate_labels):
    """
    Zero-shot classification using HuggingFace API.
    Used by intent, directness, and confrontation detection.
    """
    result = _call_api(NLI_URL, {
        "inputs": text,
        "parameters": {
            "candidate_labels": candidate_labels,
            "hypothesis_template": "This text contains {}."
        }
    })

    labels = result["labels"]
    scores = result["scores"]
    return dict(zip(labels, scores))


def compute_similarity_api(text1, text2):
    """
    Computes semantic similarity between two texts.
    Used by similarity and alignment modules.
    """
    result = _call_api(SIMILARITY_URL, {
        "inputs": {
            "source_sentence": text1,
            "sentences": [text2]
        }
    })

    # Result is a list of similarity scores
    similarity_score = round(float(result[0]), 4)
    similarity_score = max(0.0, min(1.0, similarity_score))
    return similarity_score