# pipeline_cloud.py
# Cloud version of pipeline — uses HuggingFace Inference API
# No local models needed — runs on any server with minimal RAM
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hf_inference      import (
    detect_emotion_api,
    analyze_sentiment_api,
    analyze_toxicity_api,
    classify_zero_shot_api,
    compute_similarity_api
)
from ei_engine         import ei_evaluation_engine
from feedback_module   import generate_feedback
from rephrasing_module import generate_rephrasing


# ============================================================
# INTENT DETECTION — cloud version
# ============================================================

_INTENT_HYPOTHESES = {
    "apology_detected": [
        "an apology or expression of remorse",
        "saying sorry or expressing regret"
    ],
    "blame_detected": [
        "blaming or accusing another person",
        "holding someone else responsible for a problem"
    ],
    "accountability_detected": [
        "taking personal responsibility for one's actions",
        "admitting a mistake or wrongdoing"
    ],
    "other_emotion_reference": [
        "acknowledging another person's feelings or emotions",
        "validating or recognising how someone else feels"
    ]
}

_THRESHOLD = 0.65

def _detect_signal_cloud(text, hypotheses):
    for hypothesis in hypotheses:
        scores = classify_zero_shot_api(
            text,
            [hypothesis, f"no {hypothesis}"]
        )
        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]
        if top_label == hypothesis and top_score >= _THRESHOLD:
            return True
    return False

def detect_intent_cloud(text):
    apology_detected        = _detect_signal_cloud(text, _INTENT_HYPOTHESES["apology_detected"])
    blame_detected          = _detect_signal_cloud(text, _INTENT_HYPOTHESES["blame_detected"])
    accountability_detected = _detect_signal_cloud(text, _INTENT_HYPOTHESES["accountability_detected"])
    other_emotion_reference = _detect_signal_cloud(text, _INTENT_HYPOTHESES["other_emotion_reference"])

    # Conflict resolution
    if blame_detected and accountability_detected and apology_detected:
        blame_detected = False

    return {
        "apology_detected":        apology_detected,
        "blame_detected":          blame_detected,
        "accountability_detected": accountability_detected,
        "other_emotion_reference": other_emotion_reference
    }


# ============================================================
# DIRECTNESS DETECTION — cloud version
# ============================================================

_HEDGING_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "might",
    "could", "sort of", "kind of", "i think", "i feel like",
    "i suppose", "i guess", "somewhat", "rather", "fairly",
    "a little", "a bit", "not sure", "i wonder", "seems like"
]

def _compute_hedging_score(text):
    text_lower  = text.lower()
    total_words = len(text_lower.split())
    if total_words == 0:
        return 0.5
    hedge_count    = sum(1 for h in _HEDGING_WORDS if h in text_lower)
    hedge_ratio    = hedge_count / total_words
    hedging_score  = max(0.0, 1.0 - (hedge_ratio * 8))
    return round(hedging_score, 4)

def analyze_directness_cloud(text):
    hedging_score = _compute_hedging_score(text)
    scores        = classify_zero_shot_api(text, [
        "direct and assertive communication",
        "indirect and softened communication"
    ])
    direct_score   = scores.get("direct and assertive communication", 0.5)
    indirect_score = scores.get("indirect and softened communication", 0.5)
    model_score    = round((direct_score - indirect_score + 1) / 2, 4)

    directness_score = round((0.40 * hedging_score) + (0.60 * model_score), 4)
    directness_score = max(0.0, min(1.0, directness_score))

    if directness_score >= 0.70:
        category = "High"
    elif directness_score >= 0.40:
        category = "Moderate"
    else:
        category = "Low"

    return {
        "directness_score":    directness_score,
        "directness_category": category
    }


# ============================================================
# CONFRONTATION DETECTION — cloud version
# ============================================================

import re

_ESCALATION_WORDS = [
    "always", "never", "every time", "sick of", "fed up",
    "enough", "stop it", "i am done", "forget it", "whatever",
    "fine then", "i give up", "pointless", "useless"
]

_ATTACK_PATTERNS = [
    r"\byou\b.{0,20}\balways\b",
    r"\byou\b.{0,20}\bnever\b",
    r"\byou\b.{0,20}\bfault\b",
    r"\byou\b.{0,20}\bblame\b",
]

def analyze_confrontation_cloud(text):
    text_lower = text.lower()
    total_words = len(text_lower.split())

    hit_count         = sum(1 for w in _ESCALATION_WORDS if w in text_lower)
    escalation_score  = min(1.0, (hit_count / max(total_words, 1)) * 10)

    matches       = sum(1 for p in _ATTACK_PATTERNS if re.search(p, text_lower))
    attack_score  = min(1.0, matches / 3)

    scores      = classify_zero_shot_api(text, [
        "escalating or confrontational communication",
        "calm and de-escalating communication"
    ])
    model_score = scores.get("escalating or confrontational communication", 0.5)

    confrontation_score = round(
        (0.30 * escalation_score) + (0.30 * attack_score) + (0.40 * model_score),
        4
    )
    confrontation_score = max(0.0, min(1.0, confrontation_score))

    if confrontation_score >= 0.70:
        category = "High"
    elif confrontation_score >= 0.40:
        category = "Moderate"
    else:
        category = "Low"

    return {
        "confrontation_score":    confrontation_score,
        "confrontation_category": category
    }


# ============================================================
# SIMILARITY — cloud version
# ============================================================

def analyze_similarity_cloud(scenario_text, response_text):
    similarity_score = compute_similarity_api(scenario_text, response_text)

    if similarity_score >= 0.70:
        category = "High"
    elif similarity_score >= 0.50:
        category = "Moderate"
    else:
        category = "Low"

    return {
        "similarity_score":    similarity_score,
        "similarity_category": category
    }


# ============================================================
# ALIGNMENT — cloud version
# ============================================================

_EMOTION_KEYWORDS = {
    "hurt": "sadness", "upset": "sadness", "sad": "sadness",
    "disappointed": "sadness", "devastated": "sadness",
    "heartbroken": "sadness", "let down": "sadness",
    "angry": "anger", "furious": "anger", "frustrated": "anger",
    "annoyed": "anger", "disrespected": "anger",
    "afraid": "fear", "scared": "fear", "worried": "fear",
    "anxious": "fear", "overwhelmed": "fear",
    "happy": "joy", "grateful": "joy", "excited": "joy",
    "disgusted": "disgust",
}

_EMPATHY_SIGNALS = [
    "i can see", "i understand how you feel",
    "i understand why you feel", "i understand your frustration",
    "i understand your disappointment", "i know you",
    "you must feel", "you have every right", "i hear you",
    "that must be", "you are hurt", "you are upset",
    "you feel", "i see why", "i get why", "i feel for you",
]

_EMOTION_FAMILIES = {
    "negative_heavy": ["sadness", "grief", "fear", "disgust"],
    "negative_mild":  ["anger", "surprise"],
    "positive":       ["joy", "love", "neutral"]
}

def _get_family(emotion):
    for family, emotions in _EMOTION_FAMILIES.items():
        if emotion in emotions:
            return family
    return "neutral"

def _get_scenario_emotion(scenario_text):
    text_lower = scenario_text.lower()
    for keyword, emotion in _EMOTION_KEYWORDS.items():
        if keyword in text_lower:
            return emotion
    result = detect_emotion_api(scenario_text)
    return result["emotion_label"]

def analyze_alignment_cloud(scenario_text, response_text):
    scenario_emotion = _get_scenario_emotion(scenario_text)
    response_emotion_result = detect_emotion_api(response_text)
    response_emotion = response_emotion_result["emotion_label"]

    scenario_family = _get_family(scenario_emotion)
    response_family = _get_family(response_emotion)

    if scenario_emotion == response_emotion:
        match_score = 1.0
    elif scenario_family == response_family:
        match_score = 0.7
    elif (scenario_family == "negative_heavy" and response_family == "negative_mild") or \
         (scenario_family == "negative_mild" and response_family == "negative_heavy"):
        match_score = 0.4
    elif (scenario_family == "positive" and response_family != "positive") or \
         (scenario_family != "positive" and response_family == "positive"):
        match_score = 0.1
    else:
        match_score = 0.5

    tone_similarity   = compute_similarity_api(scenario_text, response_text)
    empathy_found     = any(s in response_text.lower() for s in _EMPATHY_SIGNALS)
    neutral_scenario  = (scenario_emotion == "neutral")

    alignment_score = (0.60 * match_score) + (0.40 * tone_similarity)

    if empathy_found:
        alignment_score = max(alignment_score, 0.65)
    if neutral_scenario:
        alignment_score = max(alignment_score, 0.40)

    alignment_score = round(max(0.0, min(1.0, alignment_score)), 4)

    if alignment_score >= 0.60:
        category = "Strong"
    elif alignment_score >= 0.40:
        category = "Moderate"
    else:
        category = "Poor"

    return {
        "emotion_alignment_score": alignment_score,
        "alignment_category":      category,
        "scenario_emotion":        scenario_emotion,
        "response_emotion":        response_emotion,
        "empathy_signals_found":   empathy_found
    }


# ============================================================
# MAIN PIPELINE FUNCTION — CLOUD VERSION
# ============================================================

def analyze_ei_cloud(
    scenario_text,
    response_text,
    scenario_requires_apology,
    cultural_context="african",
    relationship_context="peer",
    power_dynamic="equal",
    stance="apologetic"
):
    # Stage 1 — NLP modules (all via HuggingFace API)
    similarity_result    = analyze_similarity_cloud(scenario_text, response_text)
    emotion_result       = detect_emotion_api(response_text)
    sentiment_result     = analyze_sentiment_api(response_text)
    toxicity_result      = analyze_toxicity_api(response_text)
    intent_result        = detect_intent_cloud(response_text)
    directness_result    = analyze_directness_cloud(response_text)
    confrontation_result = analyze_confrontation_cloud(response_text)
    alignment_result     = analyze_alignment_cloud(scenario_text, response_text)

    # Stage 2 — Extract values
    similarity_score        = similarity_result["similarity_score"]
    sentiment_score         = sentiment_result["sentiment_score"]
    toxicity_score          = toxicity_result["toxicity_score"]
    apology_detected        = intent_result["apology_detected"]
    blame_detected          = intent_result["blame_detected"]
    accountability_detected = intent_result["accountability_detected"]
    other_emotion_reference = intent_result["other_emotion_reference"]
    directness_score        = directness_result["directness_score"]
    confrontation_score     = confrontation_result["confrontation_score"]
    emotion_alignment_score = alignment_result["emotion_alignment_score"]

    # Stage 3 — EI Engine
    ei_result = ei_evaluation_engine(
        sentiment_score=sentiment_score,
        toxicity_score=toxicity_score,
        emotion_alignment_score=emotion_alignment_score,
        similarity_score=similarity_score,
        apology_detected=apology_detected,
        scenario_requires_apology=scenario_requires_apology,
        accountability_detected=accountability_detected,
        blame_detected=blame_detected,
        other_emotion_reference=other_emotion_reference,
        directness_score=directness_score,
        confrontation_score=confrontation_score,
        cultural_context=cultural_context,
        relationship_context=relationship_context,
        power_dynamic=power_dynamic
    )

    # Stage 4 — NLP outputs
    ei_result["nlp_outputs"] = {
        "similarity":    similarity_result,
        "emotion":       emotion_result,
        "sentiment":     sentiment_result,
        "toxicity":      toxicity_result,
        "intent":        intent_result,
        "directness":    directness_result,
        "confrontation": confrontation_result,
        "alignment":     alignment_result
    }

    # Stage 5 — Feedback
    feedback = generate_feedback(ei_result)
    ei_result["feedback"] = feedback

    # Stage 6 — Rephrasing
    rephrasing = generate_rephrasing(
        original_text=response_text,
        ei_result=ei_result,
        scenario_text=scenario_text,
        cultural_context=cultural_context,
        relationship_context=relationship_context,
        power_dynamic=power_dynamic,
        stance=stance
    )
    ei_result["rephrasing"] = rephrasing

    return ei_result