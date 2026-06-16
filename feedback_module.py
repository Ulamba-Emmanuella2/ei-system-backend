# feedback_module.py
# Generates human-readable feedback from EI evaluation results
#
# Input  : full ei_result dictionary from pipeline.py
# Output : structured feedback with verdict, strengths, improvements
# ============================================================


# ============================================================
# VERDICT MESSAGES
# Based on overall EI classification
# ============================================================

_VERDICTS = {
    "High Emotional Intelligence": (
        "You demonstrated strong emotional intelligence in this response. "
        "You communicated with empathy, self-awareness, and respect — "
        "the kind of response that builds trust and strengthens relationships."
    ),
    "Moderate Emotional Intelligence": (
        "You showed a reasonable level of emotional intelligence, but there "
        "is clear room to grow. Some aspects of your response were handled "
        "well, while others could be more emotionally aware and considerate."
    ),
    "Needs Improvement": (
        "Your response suggests there are some important emotional intelligence "
        "skills to develop. The good news is that EI can always be improved "
        "with awareness and practice. Let us look at what to work on."
    )
}


# ============================================================
# METRIC FEEDBACK MESSAGES
# Each metric has a positive and improvement message
# ============================================================

_METRIC_FEEDBACK = {

    "empathy": {
        "strong": "You acknowledged the other person's feelings and responded "
                  "with genuine warmth and understanding.",
        "improve": "Try to acknowledge how the other person is feeling before "
                   "expressing your own perspective. A simple 'I understand "
                   "you feel...' goes a long way."
    },

    "tone_regulation": {
        "strong": "You kept your emotional tone calm and controlled, even when "
                  "the situation was difficult.",
        "improve": "Your tone came across as harsh or emotionally charged. "
                   "Try to express your feelings without letting anger or "
                   "frustration drive your words."
    },

    "apology_appropriateness": {
        "strong": "You apologised when it was needed, which shows emotional "
                  "maturity and respect for the other person.",
        "improve": "This situation called for an apology but one was not given. "
                   "Acknowledging when you have caused hurt — even unintentionally "
                   "— is a powerful EI skill."
    },

    "context_alignment": {
        "strong": "Your response directly addressed the situation at hand, "
                  "showing that you understood what was really going on.",
        "improve": "Your response did not fully address the situation. Make sure "
                   "you are responding to what actually happened, not deflecting "
                   "or changing the subject."
    },

    "self_awareness": {
        "strong": "You took responsibility for your role in the situation without "
                  "shifting blame onto others — a sign of real self-awareness.",
        "improve": "Try to reflect on your own role before responding. Shifting "
                   "blame onto others — even partially — makes it harder to "
                   "resolve conflict and damages trust."
    },

    "perspective_taking": {
        "strong": "You showed an ability to see the situation from the other "
                  "person's point of view, which is at the heart of emotional "
                  "intelligence.",
        "improve": "Practice putting yourself in the other person's shoes. Ask "
                   "yourself: how would I feel if I were them in this situation? "
                   "Let that guide your response."
    }
}


# ============================================================
# CULTURAL FEEDBACK ADDITIONS
# Extra advice based on cultural context and relationship
# ============================================================

_CULTURAL_NOTES = {
    "african": {
        "elder": {
            "directness": "When speaking with an elder, direct or blunt language "
                          "can come across as disrespectful, even if unintentional. "
                          "A softer, more deferential tone shows cultural awareness.",
            "confrontation": "Confronting or arguing with an elder is considered "
                             "a serious social violation in most African contexts. "
                             "Express disagreement indirectly and with great respect."
        },
        "peer": {
            "directness": "With peers in African contexts, moderate directness is "
                          "generally acceptable, but warmth and relationship should "
                          "still come before bluntness.",
            "confrontation": "Even with peers, high confrontation can damage the "
                             "relationship long term. Aim to de-escalate rather "
                             "than match aggression with aggression."
        }
    },
    "western": {
        "peer": {
            "directness": "Direct communication is valued in Western contexts — "
                          "being clear and honest is seen as respectful.",
            "confrontation": "While directness is acceptable, confrontational "
                             "language still damages relationships. Stay assertive "
                             "without being aggressive."
        }
    },
    "asian": {
        "elder": {
            "directness": "In Asian cultural contexts, indirect communication with "
                          "elders is strongly preferred. Blunt language can cause "
                          "significant loss of face for both parties.",
            "confrontation": "Confrontation with elders in Asian contexts is a "
                             "serious cultural violation. Harmony and face-saving "
                             "should always guide your response."
        }
    }
}


# ============================================================
# THRESHOLD — what counts as strong vs needs improvement
# ============================================================

_STRONG_THRESHOLD  = 75   # score >= 70 → mention as strength
_IMPROVE_THRESHOLD = 60   # score <  55 → mention as improvement area


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_strengths(metrics):
    """Returns list of metrics that scored strongly."""
    return [
        metric for metric, score in metrics.items()
        if score >= _STRONG_THRESHOLD
    ]


def _get_improvements(metrics):
    """
    Returns list of metrics that need improvement,
    sorted from lowest to highest score.
    Max 3 improvement areas — don't overwhelm the user.
    """
    weak = [
        (metric, score) for metric, score in metrics.items()
        if score < _IMPROVE_THRESHOLD
    ]
    weak_sorted = sorted(weak, key=lambda x: x[1])
    return [metric for metric, score in weak_sorted[:3]]


def _get_cultural_note(cultural_context, relationship_context, categories):
    """
    Returns a relevant cultural note if directness or
    confrontation was an issue in this cultural context.
    """
    notes = []

    ctx  = cultural_context.lower()
    rel  = relationship_context.lower()

    if ctx not in _CULTURAL_NOTES:
        return notes
    if rel not in _CULTURAL_NOTES[ctx]:
        rel = "peer"   # fallback to peer if relationship not defined
    if rel not in _CULTURAL_NOTES[ctx]:
        return notes

    context_notes = _CULTURAL_NOTES[ctx][rel]

    # Add directness note if directness was High in a sensitive context
    if categories.get("directness") == "High":
        if "directness" in context_notes:
            notes.append(context_notes["directness"])

    # Add confrontation note if confrontation was High or Moderate
    if categories.get("confrontation") in ["High", "Moderate"]:
        if "confrontation" in context_notes:
            notes.append(context_notes["confrontation"])

    return notes


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_feedback(ei_result):
    """
    Generates personalised human-readable feedback
    from the EI evaluation result.

    Parameters:
        ei_result (dict) — full result from pipeline.analyze_ei()

    Output:
        dict with keys:
            verdict      (str)  — overall summary
            strengths    (list) — what they did well
            improvements (list) — what to work on
            cultural     (list) — cultural context advice
            full_text    (str)  — everything combined into one message
    """

    classification   = ei_result["classification"]
    metrics          = ei_result["metrics"]
    categories       = ei_result["categories"]
    context          = ei_result["context"]
    ei_score         = ei_result["ei_score"]

    cultural_context     = context.get("cultural_context", "african")
    relationship_context = context.get("relationship_context", "peer")

    # ----------------------------------------------------------
    # 1. Verdict
    # ----------------------------------------------------------
    verdict = _VERDICTS.get(classification, _VERDICTS["Moderate Emotional Intelligence"])

    # ----------------------------------------------------------
    # 2. Strengths
    # ----------------------------------------------------------
    strong_metrics = _get_strengths(metrics)
    strengths = [
        _METRIC_FEEDBACK[metric]["strong"]
        for metric in strong_metrics
        if metric in _METRIC_FEEDBACK
    ]

    # ----------------------------------------------------------
    # 3. Improvements
    # ----------------------------------------------------------
    weak_metrics = _get_improvements(metrics)
    improvements = [
        _METRIC_FEEDBACK[metric]["improve"]
        for metric in weak_metrics
        if metric in _METRIC_FEEDBACK
    ]

    # ----------------------------------------------------------
    # 4. Cultural notes
    # ----------------------------------------------------------
    cultural_notes = _get_cultural_note(
        cultural_context,
        relationship_context,
        categories
    )

    # ----------------------------------------------------------
    # 5. Combine into full text
    # ----------------------------------------------------------
    lines = []

    lines.append(f"EI Score: {ei_score}/100 — {classification}")
    lines.append("")
    lines.append(verdict)

    if strengths:
        lines.append("")
        lines.append("What you did well:")
        for s in strengths:
            lines.append(f"  + {s}")

    if improvements:
        lines.append("")
        lines.append("What to work on:")
        for i in improvements:
            lines.append(f"  - {i}")

    if cultural_notes:
        lines.append("")
        lines.append("Cultural awareness:")
        for note in cultural_notes:
            lines.append(f"  * {note}")

    full_text = "\n".join(lines)

    return {
        "verdict":      verdict,
        "strengths":    strengths,
        "improvements": improvements,
        "cultural":     cultural_notes,
        "full_text":    full_text
    }


# ============================================================
# TESTS — Run: !python feedback_module.py
# ============================================================

def print_feedback(label, feedback):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    print(feedback["full_text"])


if __name__ == "__main__":

    # Simulate EI results directly without running the full pipeline
    # This lets you test feedback_module independently

    # TEST 1 — High EI result
    high_ei_result = {
        "ei_score": 88.7,
        "classification": "High Emotional Intelligence",
        "metrics": {
            "empathy":                 94.5,
            "tone_regulation":         85.0,
            "apology_appropriateness": 90.0,
            "context_alignment":       90.0,
            "self_awareness":          85.0,
            "perspective_taking":      85.0
        },
        "categories": {
            "sentiment":        "Positive",
            "toxicity":         "Low",
            "emotion_alignment": "Strong",
            "similarity":       "High",
            "directness":       "Moderate",
            "confrontation":    "Low"
        },
        "context": {
            "cultural_context":     "african",
            "relationship_context": "peer",
            "power_dynamic":        "equal"
        }
    }
    print_feedback("High EI — Peer (African)", generate_feedback(high_ei_result))

    # TEST 2 — Low EI result
    low_ei_result = {
        "ei_score": 27.1,
        "classification": "Needs Improvement",
        "metrics": {
            "empathy":                 36.75,
            "tone_regulation":         15.0,
            "apology_appropriateness":  0.0,
            "context_alignment":       35.0,
            "self_awareness":          30.0,
            "perspective_taking":      40.0
        },
        "categories": {
            "sentiment":        "Negative",
            "toxicity":         "High",
            "emotion_alignment": "Poor",
            "similarity":       "Low",
            "directness":       "High",
            "confrontation":    "High"
        },
        "context": {
            "cultural_context":     "african",
            "relationship_context": "elder",
            "power_dynamic":        "higher"
        }
    }
    print_feedback("Low EI — Elder (African)", generate_feedback(low_ei_result))

    # TEST 3 — Moderate EI result
    moderate_ei_result = {
        "ei_score": 58.1,
        "classification": "Moderate Emotional Intelligence",
        "metrics": {
            "empathy":                 68.25,
            "tone_regulation":         60.0,
            "apology_appropriateness": 50.0,
            "context_alignment":       65.0,
            "self_awareness":          60.0,
            "perspective_taking":      40.0
        },
        "categories": {
            "sentiment":        "Neutral",
            "toxicity":         "Moderate",
            "emotion_alignment": "Moderate",
            "similarity":       "Moderate",
            "directness":       "Moderate",
            "confrontation":    "Moderate"
        },
        "context": {
            "cultural_context":     "african",
            "relationship_context": "peer",
            "power_dynamic":        "equal"
        }
    }
    print_feedback("Moderate EI — Peer (African)", generate_feedback(moderate_ei_result))