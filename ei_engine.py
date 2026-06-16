# ei_engine.py
# EI Evaluation Engine — with full context-based cultural adaptation

# ============================================================
# STEP 1: CATEGORIZATION FUNCTIONS
# ============================================================

def categorize_sentiment(sentiment_score):
    """
    sentiment_score ∈ [-1, 1]
    Returns: "Positive", "Negative", or "Neutral"
    """
    if sentiment_score <= -0.2:
        return "Negative"
    elif sentiment_score >= 0.2:
        return "Positive"
    else:
        return "Neutral"


def categorize_toxicity(toxicity_score):
    """
    toxicity_score ∈ [0, 1]
    Returns: "Low", "Moderate", or "High"
    """
    if toxicity_score <= 0.30:
        return "Low"
    elif toxicity_score <= 0.50:
        return "Moderate"
    else:
        return "High"


def categorize_emotion_alignment(emotion_alignment_score):
    """
    emotion_alignment_score ∈ [0, 1]
    Returns: "Strong", "Moderate", or "Poor"
    """
    if emotion_alignment_score >= 0.60:
        return "Strong"
    elif emotion_alignment_score >= 0.40:
        return "Moderate"
    else:
        return "Poor"


def categorize_similarity(similarity_score):
    """
    similarity_score ∈ [0, 1]
    Returns: "High", "Moderate", or "Low"
    """
    if similarity_score >= 0.70:
        return "High"
    elif similarity_score >= 0.50:
        return "Moderate"
    else:
        return "Low"


def categorize_directness(directness_score):
    """
    directness_score ∈ [0, 1]
    Returns: "High", "Moderate", or "Low"
    """
    if directness_score >= 0.70:
        return "High"
    elif directness_score >= 0.40:
        return "Moderate"
    else:
        return "Low"


def categorize_confrontation(confrontation_score):
    """
    confrontation_score ∈ [0, 1]
    Returns: "High", "Moderate", or "Low"
    """
    if confrontation_score >= 0.70:
        return "High"
    elif confrontation_score >= 0.40:
        return "Moderate"
    else:
        return "Low"


# ============================================================
# STEP 2: EI METRIC FUNCTIONS (exact lookup values)
# ============================================================

def compute_empathy(alignment_level, toxicity_level):
    """
    Uses alignment + toxicity levels to assign empathy score.

    Strong alignment + Low toxicity = 90  (person is in tune and calm)
    Moderate alignment              = 65  (some emotional awareness)
    Anything else                   = 35  (poor emotional connection)
    """
    if alignment_level == "Strong" and toxicity_level == "Low":
        empathy = 90
    elif alignment_level == "Moderate":
        empathy = 65
    else:
        empathy = 35

    return empathy


def compute_tone(toxicity_level, sentiment_level):
    """
    Uses toxicity level as base, with small bonus for
    negative-but-not-toxic responses (shows emotional honesty).

    High toxicity   = 25  (tone is harmful)
    Moderate        = 60  (tone is acceptable)
    Low toxicity    = 85  (tone is well regulated)
    Bonus +5 if sentiment is negative but toxicity is still low
    → shows the person is honest about feelings without being aggressive
    """
    if toxicity_level == "High":
        tone = 25
    elif toxicity_level == "Moderate":
        tone = 60
    else:
        tone = 85

    if sentiment_level == "Negative" and toxicity_level == "Low":
        tone += 5

    return tone


def compute_apology(apology_detected, scenario_requires_apology):
    """
    Apology is only judged when the scenario requires it.

    Required + given     = 90  (did the right thing)
    Required + not given = 0   (failed to repair)
    Not required         = 50  (neutral — not applicable)
    """
    if scenario_requires_apology:
        if apology_detected:
            apology = 90
        else:
            apology = 0
    else:
        apology = 50

    return apology


def compute_context_alignment(similarity_level):
    """
    How relevant was the response to the situation?

    High similarity     = 90  (response directly addressed the situation)
    Moderate similarity = 65  (partially relevant)
    Low similarity      = 35  (response missed the point)
    """
    if similarity_level == "High":
        context_alignment = 90
    elif similarity_level == "Moderate":
        context_alignment = 65
    else:
        context_alignment = 35

    return context_alignment


def compute_self_awareness(accountability_detected, blame_detected):
    """
    Accountability raises score, blame lowers it significantly.

    Accountable + no blame = 85  (owns their role)
    Blame detected         = 30  (deflecting responsibility)
    Neither                = 60  (passive — didn't engage either way)
    """
    if accountability_detected and not blame_detected:
        self_awareness = 85
    elif blame_detected:
        self_awareness = 30
    else:
        self_awareness = 60

    return self_awareness


def compute_perspective_taking(other_emotion_reference, toxicity_level, blame_detected):
    """
    Did the user acknowledge the other person's feelings?
    Perspective taking requires genuine acknowledgement —
    not just mentioning the other person in an accusation.

    Genuine acknowledgement + no blame + low toxicity = 85
    Acknowledgement but blaming or toxic                = 50
    No acknowledgement                                  = 40
    """
    if other_emotion_reference:
        if blame_detected or toxicity_level in ["Moderate", "High"]:
            perspective_taking = 50   # mentioned feelings but still attacking
        else:
            perspective_taking = 85   # genuine acknowledgement
    else:
        perspective_taking = 40

    return perspective_taking


# ============================================================
# STEP 3: CULTURAL ADAPTATION (context-based)
#
# Two layers work together:
#   cultural_context    → the cultural background (african/western/asian)
#   relationship_context → who they are talking to (elder/peer/subordinate/stranger)
#   power_dynamic       → authority gap (higher/equal/lower)
#
# Key insight: The same words mean different things
# depending on WHO you say them to.
# ============================================================

def apply_cultural_adaptation(
    cultural_context,
    relationship_context,
    power_dynamic,
    directness_level,
    confrontation_level,
    blame_detected,
    apology_detected,
    scenario_requires_apology,
    empathy,
    tone,
    apology,
    self_awareness,
    perspective_taking
):
    """
    Adjusts EI metric scores based on cultural communication norms
    AND the relationship between the speaker and the listener.
    """
    cultural_context     = cultural_context.lower()
    relationship_context = relationship_context.lower()
    power_dynamic        = power_dynamic.lower()

    # Determine if speaking to someone with higher authority
    talking_to_authority = (
        relationship_context == "elder" or power_dynamic == "higher"
    )

    # ----------------------------------------------------------
    # AFRICAN CONTEXT
    # Indirect communication is preferred, especially with elders.
    # Confrontation and bluntness are signs of poor upbringing.
    # Empathy and humility are highly valued.
    # ----------------------------------------------------------
    if cultural_context == "african":

        if talking_to_authority:
            # Being highly direct with an elder is disrespectful
            if directness_level == "High":
                tone -= 20              # bigger penalty than with peers

            # Confronting an elder is a serious cultural violation
            if confrontation_level in ["High", "Moderate"]:
                tone -= 15

            # Blaming an elder shows very poor EI and cultural awareness
            if blame_detected:
                self_awareness -= 20

            # Apologising to an elder when needed shows maturity and humility
            if apology_detected and scenario_requires_apology:
                apology += 10           # extra cultural credit

            # Empathy matters even more when speaking to an elder
            empathy *= 1.10

        elif relationship_context == "peer":
            # Moderate penalty for high directness with peers
            if directness_level == "High":
                tone -= 10
            empathy *= 1.05

        elif relationship_context == "subordinate":
            # Directness is more acceptable with a subordinate
            if directness_level == "High":
                tone -= 5
            # But blaming a subordinate is still poor EI
            if blame_detected:
                self_awareness -= 15
            empathy *= 1.05

        else:  # stranger or unspecified
            if directness_level == "High":
                tone -= 10
            empathy *= 1.05

    # ----------------------------------------------------------
    # WESTERN CONTEXT
    # Direct communication is generally acceptable and even valued.
    # Self-awareness and personal accountability are culturally important.
    # Authority relationships are less rigid than African/Asian contexts.
    # ----------------------------------------------------------
    elif cultural_context == "western":

        if talking_to_authority:
            # Slight penalty for being too direct — some hierarchy still exists
            if directness_level == "High":
                tone -= 5
            self_awareness *= 1.05

        elif relationship_context == "peer":
            # Directness with peers is rewarded in Western culture
            if directness_level == "High":
                tone += 5
            self_awareness *= 1.05

        elif relationship_context == "subordinate":
            # Directness acceptable with subordinates too
            if directness_level == "High":
                tone += 5
            # But blaming a subordinate is a leadership failure
            if blame_detected:
                self_awareness -= 20

    # ----------------------------------------------------------
    # ASIAN CONTEXT
    # Harmony and face-saving are deeply valued.
    # Confrontation is especially inappropriate with authority figures.
    # Perspective-taking is culturally amplified.
    # ----------------------------------------------------------
    elif cultural_context == "asian":

        if talking_to_authority:
            # Confronting an elder or superior is a very serious violation
            if confrontation_level == "High":
                tone -= 25
            # High directness is also inappropriate
            if directness_level == "High":
                tone -= 15
            # Perspective-taking is especially valued in this context
            perspective_taking *= 1.15

        elif relationship_context == "peer":
            if confrontation_level == "High":
                tone -= 15
            perspective_taking *= 1.10

        elif relationship_context == "subordinate":
            if confrontation_level == "High":
                tone -= 10
            perspective_taking *= 1.05

    return empathy, tone, apology, self_awareness, perspective_taking


# ============================================================
# STEP 4: NORMALIZATION
# ============================================================

def normalize(score):
    """Clamps any score to [0, 100]"""
    return max(0, min(100, score))


# ============================================================
# MAIN ENGINE FUNCTION
# ============================================================

def ei_evaluation_engine(
    sentiment_score,
    toxicity_score,
    emotion_alignment_score,
    similarity_score,
    apology_detected,
    scenario_requires_apology,
    accountability_detected,
    blame_detected,
    other_emotion_reference,
    directness_score,
    confrontation_score,
    cultural_context="african",
    relationship_context="peer",        # NEW: elder / peer / subordinate / stranger
    power_dynamic="equal"               # NEW: higher / equal / lower
):
    """
    Full EI Evaluation Engine.

    Pipeline:
    Categorize → Compute Metrics → Cultural Adapt → Normalize → Score → Classify
    """

    # ----------------------------------------------------------
    # STEP 1: Categorize all raw scores into levels`
    # ----------------------------------------------------------
    sentiment_level     = categorize_sentiment(sentiment_score)
    toxicity_level      = categorize_toxicity(toxicity_score)
    alignment_level     = categorize_emotion_alignment(emotion_alignment_score)
    similarity_level    = categorize_similarity(similarity_score)
    directness_level    = categorize_directness(directness_score)
    confrontation_level = categorize_confrontation(confrontation_score)

    # ----------------------------------------------------------
    # STEP 2: Compute raw EI metric scores (lookup-based)
    # ----------------------------------------------------------
    empathy            = compute_empathy(alignment_level, toxicity_level)
    tone               = compute_tone(toxicity_level, sentiment_level)
    apology            = compute_apology(apology_detected, scenario_requires_apology)
    context_alignment  = compute_context_alignment(similarity_level)
    self_awareness     = compute_self_awareness(accountability_detected, blame_detected)
    perspective_taking = compute_perspective_taking(other_emotion_reference, toxicity_level, blame_detected)

    # ----------------------------------------------------------
    # STEP 3: Apply cultural adaptation
    # ----------------------------------------------------------
    empathy, tone, apology, self_awareness, perspective_taking = apply_cultural_adaptation(
        cultural_context,
        relationship_context,
        power_dynamic,
        directness_level,
        confrontation_level,
        blame_detected,
        apology_detected,
        scenario_requires_apology,
        empathy,
        tone,
        apology,
        self_awareness,
        perspective_taking
    )

    # ----------------------------------------------------------
    # STEP 4: Normalize all scores to [0, 100]
    # ----------------------------------------------------------
    empathy            = normalize(empathy)
    tone               = normalize(tone)
    apology            = normalize(apology)
    context_alignment  = normalize(context_alignment)
    self_awareness     = normalize(self_awareness)
    perspective_taking = normalize(perspective_taking)

    # ----------------------------------------------------------
    # STEP 5: Compute weighted final EI score
    # ----------------------------------------------------------
    ei_score = (
        0.20 * empathy +
        0.15 * tone +
        0.15 * apology +
        0.20 * context_alignment +
        0.15 * self_awareness +
        0.15 * perspective_taking
    )
    ei_score = round(ei_score, 1)

    # ----------------------------------------------------------
    # STEP 6: Classify
    # ----------------------------------------------------------
    if ei_score >= 80:
        classification = "High Emotional Intelligence"
    elif ei_score >= 50:
        classification = "Moderate Emotional Intelligence"
    else:
        classification = "Needs Improvement"

    # ----------------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------------
    return {
        "ei_score": ei_score,
        "classification": classification,
        "metrics": {
            "empathy":                 empathy,
            "tone_regulation":         tone,
            "apology_appropriateness": apology,
            "context_alignment":       context_alignment,
            "self_awareness":          self_awareness,
            "perspective_taking":      perspective_taking
        },
        "categories": {
            "sentiment":          sentiment_level,
            "toxicity":           toxicity_level,
            "emotion_alignment":  alignment_level,
            "similarity":         similarity_level,
            "directness":         directness_level,
            "confrontation":      confrontation_level
        },
        "context": {
            "cultural_context":     cultural_context,
            "relationship_context": relationship_context,
            "power_dynamic":        power_dynamic
        }
    }


# ============================================================
# TESTS — Run: python ei_engine.py
# ============================================================

def print_result(label, result):
    print(f"\n{'='*55}")
    print(f"TEST: {label}")
    print(f"{'='*55}")
    print(f"EI Score      : {result['ei_score']}")
    print(f"Classification: {result['classification']}")
    print(f"\n--- Context ---")
    for k, v in result["context"].items():
        print(f"  {k:<28}: {v}")
    print(f"\n--- Metric Scores ---")
    for metric, score in result["metrics"].items():
        print(f"  {metric:<28}: {score}")
    print(f"\n--- Input Categories ---")
    for cat, level in result["categories"].items():
        print(f"  {cat:<28}: {level}")


if __name__ == "__main__":

    # ----------------------------------------------------------
    # TEST 1: High EI — Ideal empathetic response (peer, african)
    # ----------------------------------------------------------
    r1 = ei_evaluation_engine(
        sentiment_score=0.75,
        toxicity_score=0.05,
        emotion_alignment_score=0.80,
        similarity_score=0.85,
        apology_detected=True,
        scenario_requires_apology=True,
        accountability_detected=True,
        blame_detected=False,
        other_emotion_reference=True,
        directness_score=0.50,
        confrontation_score=0.10,
        cultural_context="african",
        relationship_context="peer",
        power_dynamic="equal"
    )
    print_result("High EI — Ideal Empathetic Response (Peer)", r1)

    # ----------------------------------------------------------
    # TEST 2: Low EI — Toxic blaming response (peer, african)
    # ----------------------------------------------------------
    r2 = ei_evaluation_engine(
        sentiment_score=-0.85,
        toxicity_score=0.75,
        emotion_alignment_score=0.20,
        similarity_score=0.30,
        apology_detected=False,
        scenario_requires_apology=True,
        accountability_detected=False,
        blame_detected=True,
        other_emotion_reference=False,
        directness_score=0.90,
        confrontation_score=0.85,
        cultural_context="african",
        relationship_context="peer",
        power_dynamic="equal"
    )
    print_result("Low EI — Toxic Blaming Response (Peer)", r2)

    # ----------------------------------------------------------
    # TEST 3: Moderate EI — Mixed signals (peer, african)
    # ----------------------------------------------------------
    r3 = ei_evaluation_engine(
        sentiment_score=0.10,
        toxicity_score=0.35,
        emotion_alignment_score=0.50,
        similarity_score=0.60,
        apology_detected=False,
        scenario_requires_apology=False,
        accountability_detected=False,
        blame_detected=False,
        other_emotion_reference=False,
        directness_score=0.55,
        confrontation_score=0.45,
        cultural_context="african",
        relationship_context="peer",
        power_dynamic="equal"
    )
    print_result("Moderate EI — Mixed Signals (Peer)", r3)

    # ----------------------------------------------------------
    # TEST 4: CULTURAL COMPARISON
    # Same blunt, confrontational, blaming response —
    # directed at an ELDER vs a PEER in African context.
    # Elder score should be noticeably lower.
    # ----------------------------------------------------------
    base_inputs = dict(
        sentiment_score=0.10,
        toxicity_score=0.25,
        emotion_alignment_score=0.55,
        similarity_score=0.60,
        apology_detected=False,
        scenario_requires_apology=True,
        accountability_detected=False,
        blame_detected=True,
        other_emotion_reference=False,
        directness_score=0.80,
        confrontation_score=0.75,
        cultural_context="african"
    )

    r4_elder = ei_evaluation_engine(
        **base_inputs,
        relationship_context="elder",
        power_dynamic="higher"
    )
    r4_peer = ei_evaluation_engine(
        **base_inputs,
        relationship_context="peer",
        power_dynamic="equal"
    )

    print_result("Cultural Test — Same Response to an ELDER (African)", r4_elder)
    print_result("Cultural Test — Same Response to a PEER (African)", r4_peer)

    # ----------------------------------------------------------
    # TEST 5: AFRICAN ELDER — Respectful, humble response
    # Should score very high because of cultural bonus
    # ----------------------------------------------------------
    r5 = ei_evaluation_engine(
        sentiment_score=0.65,
        toxicity_score=0.05,
        emotion_alignment_score=0.75,
        similarity_score=0.80,
        apology_detected=True,
        scenario_requires_apology=True,
        accountability_detected=True,
        blame_detected=False,
        other_emotion_reference=True,
        directness_score=0.30,          # indirect — respectful
        confrontation_score=0.05,       # very low confrontation
        cultural_context="african",
        relationship_context="elder",
        power_dynamic="higher"
    )
    print_result("High EI — Respectful Response to Elder (African)", r5)