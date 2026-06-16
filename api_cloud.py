# api_cloud.py
# Cloud version of Flask API
# Uses HuggingFace Inference API — no local models needed
# ============================================================

import os
from flask import Flask, request, jsonify
from pipeline_cloud import analyze_ei_cloud
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "running",
        "message": "EI system API is online (cloud version)"
    }), 200

@app.route("/test-all", methods=["GET"])
def test_all():
    import requests, os
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://router.huggingface.co/hf-inference/models"
    results = {}

    # Test sentiment
    try:
        r = requests.post(f"{base}/cardiffnlp/twitter-roberta-base-sentiment-latest",
            headers=headers, json={"inputs": "I am sorry"}, timeout=30)
        results["sentiment"] = r.json()
    except Exception as e:
        results["sentiment"] = str(e)

    # Test toxicity
    try:
        r = requests.post(f"{base}/unitary/toxic-bert",
            headers=headers, json={"inputs": "I am sorry"}, timeout=30)
        results["toxicity"] = r.json()
    except Exception as e:
        results["toxicity"] = str(e)

    # Test NLI
    try:
        r = requests.post(f"{base}/facebook/bart-large-mnli",
            headers=headers, json={"inputs": "I am sorry", "parameters": {"candidate_labels": ["apology", "denial"]}}, timeout=30)
        results["nli"] = r.json()
    except Exception as e:
        results["nli"] = str(e)

    return jsonify(results)

@app.route("/test-emotion", methods=["GET"])
def test_emotion():
    import requests, os
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/j-hartmann/emotion-english-distilroberta-base",
            headers=headers,
            json={"inputs": "I am really sorry I hurt you"},
            timeout=30
        )
        return jsonify({"status": r.status_code, "response": r.json()})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/test-hf-router", methods=["GET"])
def test_hf_router():
    import requests, os
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
            headers=headers,
            json={"inputs": {"source_sentence": "hello", "sentences": ["hi"]}},
            timeout=30
        )
        return jsonify({"status": r.status_code, "response": r.json()})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/test-hf-api", methods=["GET"])
def test_hf_api():
    import requests, os
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            headers=headers,
            json={"inputs": {"source_sentence": "hello", "sentences": ["hi"]}},
            timeout=30
        )
        return jsonify({"status": r.status_code, "response": r.json()})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/test-hf", methods=["GET"])
def test_hf():
    import requests
    try:
        r = requests.get("https://huggingface.co", timeout=10)
        return jsonify({"status": "reachable", "code": r.status_code})
    except Exception as e:
        return jsonify({"status": "unreachable", "error": str(e)})

@app.route("/debug", methods=["GET"])
def debug():
    import os
    token = os.environ.get("HF_TOKEN", "")
    return jsonify({
        "token_set": bool(token),
        "token_preview": token[:8] + "..." if token else "EMPTY"
    }), 200
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    required = ["scenario_text", "response_text", "scenario_requires_apology"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    scenario_text             = data["scenario_text"]
    response_text             = data["response_text"]
    scenario_requires_apology = bool(data["scenario_requires_apology"])
    cultural_context          = data.get("cultural_context",     "african")
    relationship_context      = data.get("relationship_context", "peer")
    power_dynamic             = data.get("power_dynamic",        "equal")
    stance                    = data.get("stance",               "apologetic")

    valid_stances = ["apologetic", "assertive", "neutral"]
    if stance not in valid_stances:
        stance = "apologetic"

    if not scenario_text.strip():
        return jsonify({"error": "scenario_text cannot be empty"}), 400
    if not response_text.strip():
        return jsonify({"error": "response_text cannot be empty"}), 400

    try:
        ei_result = analyze_ei_cloud(
            scenario_text=scenario_text,
            response_text=response_text,
            scenario_requires_apology=scenario_requires_apology,
            cultural_context=cultural_context,
            relationship_context=relationship_context,
            power_dynamic=power_dynamic,
            stance=stance
        )
    except Exception as e:
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500

    response = {
        "ei_score":       ei_result["ei_score"],
        "classification": ei_result["classification"],
        "metrics":        ei_result["metrics"],
        "categories":     ei_result["categories"],
        "context":        ei_result["context"],
        "feedback":       ei_result.get("feedback",   {}),
        "rephrasing":     ei_result.get("rephrasing", {})
    }

    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)