# api_cloud.py
# Cloud version of Flask API
# Uses HuggingFace Inference API — no local models needed
# ============================================================

import os
import requests
from flask import Flask, request, jsonify
from pipeline_cloud import analyze_ei_cloud
from session_manager import start_session, process_reply, end_session, get_session

RECAPTCHA_SECRET = os.environ.get("RECAPTCHA_SECRET_KEY", "")
app = Flask(__name__)

def verify_recaptcha(token):
    response = requests.post(
        "https://www.google.com/recaptcha/api/siteverify",
        data={
            "secret": RECAPTCHA_SECRET,
            "response": token
        }
    )
    result = response.json()
    return result.get("success", False)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "running",
        "message": "EI system API is online (cloud version)"
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    token = data.get("recaptcha_token")
    if not verify_recaptcha(token):
        return jsonify({"error": "reCAPTCHA verification failed"}), 403

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


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    token = data.get("recaptcha_token")
    if not verify_recaptcha(token):
        return jsonify({"error": "reCAPTCHA verification failed"}), 403

    try:
        result = start_session(
            situation=data["situation"],
            cultural_context=data.get("cultural_context", "african"),
            relationship_context=data.get("relationship_context", "peer"),
            power_dynamic=data.get("power_dynamic", "equal"),
            goal=data.get("goal", "apologise")
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reply", methods=["POST"])
def reply():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
    try:
        result = process_reply(
            session_id=data["session_id"],
            user_message=data["user_message"]
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/end", methods=["POST"])
def end():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
    try:
        result = end_session(session_id=data["session_id"])
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)