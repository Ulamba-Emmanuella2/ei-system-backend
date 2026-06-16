# rephrasing_module.py
# Generates emotionally intelligent rephrasing of user responses
# Uses rule-based template generation — no API key required
#
# Stance options:
#   apologetic → person was wrong, needs to repair
#   assertive  → person was right, needs to hold firm with empathy
#   neutral    → mixed situation, needs dialogue
#
# Input  : original text + ei_result + context + stance
# Output : rephrased text + explanation of changes
# ============================================================


# ============================================================
# APOLOGETIC TEMPLATES
# For situations where the person was wrong and needs to repair
# ============================================================

# Opening phrases — acknowledge the situation
_OPENINGS = {
    "apology_needed": [
        "I am truly sorry for what happened.",
        "I sincerely apologise for my actions.",
        "I want to start by saying I am sorry.",
        "Please accept my sincere apology.",
    ],
    "accountability": [
        "I take full responsibility for this.",
        "I recognise that I was wrong in how I handled this.",
        "Looking back, I should have done better.",
        "I own my part in this situation.",
    ],
    "empathy": [
        "I can see that this has really affected you.",
        "I understand how hurt you must be feeling.",
        "I can only imagine how difficult this has been for you.",
        "Your feelings make complete sense given what happened.",
    ],
    "neutral": [
        "I have been thinking about what happened.",
        "I want to address this properly.",
        "I think it is important we talk about this.",
    ]
}

# Elder specific openings — more humble and deferential
_ELDER_OPENINGS = [
    "With the greatest respect,",
    "I humbly come before you,",
    "Please forgive me,",
    "With all due respect,",
]

# Middle phrases — address the core issue
_MIDDLES = {
    "blame_removed": [
        "I know that my actions caused this situation.",
        "I should have handled things differently.",
        "This happened because of choices I made.",
    ],
    "emotion_acknowledged": [
        "Your feelings are completely valid.",
        "I hear you and I understand your frustration.",
        "It makes sense that you feel this way.",
    ],
    "resolution": [
        "I would like to make this right.",
        "I want to find a way to move forward together.",
        "I am committed to doing better.",
    ]
}

# Closing phrases — forward looking
_CLOSINGS = {
    "peer": [
        "Can we talk about how to resolve this together?",
        "I value our relationship and want to make this right.",
        "I hope we can work through this.",
    ],
    "elder": [
        "I ask for your guidance and forgiveness.",
        "I will do better and I ask for your patience.",
        "Please know that I deeply respect you and value your wisdom.",
    ],
    "subordinate": [
        "I am open to discussing how to move forward.",
        "I want to support you through this.",
        "Please let me know how I can make this right.",
    ],
    "stranger": [
        "I hope we can resolve this respectfully.",
        "I am open to finding a solution together.",
    ]
}


# ============================================================
# ASSERTIVE TEMPLATES
# For situations where the person made the RIGHT decision
# but needs to hold firm while remaining empathetic.
#
# This represents one of the highest EI skills —
# empathetic assertiveness. Being emotionally intelligent
# does NOT mean always apologising or backing down.
# Sometimes the right response is:
# "I understand how you feel AND I am standing by my decision."
# ============================================================

_ASSERTIVE_OPENINGS = {
    "peer": [
        "I hear you and I understand this has been difficult.",
        "I can see this has affected you and I want to acknowledge that.",
        "I genuinely understand why you feel this way.",
    ],
    "elder": [
        "With the greatest respect, I hear your concerns and they matter to me.",
        "I deeply value your opinion and I understand your disappointment.",
        "I love and respect you deeply, and I hear what you are saying.",
    ],
    "subordinate": [
        "I understand this decision has affected you and I want to acknowledge that.",
        "I hear your concerns and they are valid.",
    ],
    "stranger": [
        "I understand this is difficult and I want to acknowledge your feelings.",
        "I hear what you are saying and I appreciate you sharing that.",
    ]
}

_ASSERTIVE_MIDDLES = [
    "At the same time, this was a decision I made carefully and thoughtfully.",
    "I want you to know that I did not take this lightly.",
    "I considered this carefully before moving forward.",
    "This was not an easy decision but it was the right one for me.",
]

_ASSERTIVE_CLOSINGS = {
    "peer": [
        "I hope we can find a way to understand each other even if we see this differently.",
        "I value our relationship and I hope we can move forward with mutual respect.",
    ],
    "elder": [
        "I hope with time you will understand and I ask for your continued love and support.",
        "Your support means everything to me and I hope we can work through this together.",
        "I remain committed to this family and I ask for your patience and understanding.",
    ],
    "subordinate": [
        "I am open to discussing how this affects you and how we can support each other.",
        "I want to make sure you feel heard even as we move forward with this decision.",
    ],
    "stranger": [
        "I hope we can find a way to respect each other's positions.",
        "I remain open to discussion while standing by what I believe is right.",
    ]
}


# ============================================================
# NEUTRAL TEMPLATES
# For mixed situations — partial acknowledgment
# Neither fully apologetic nor fully assertive
# Used when blame is unclear or situation is a misunderstanding
# ============================================================

_NEUTRAL_OPENINGS = [
    "I have been reflecting on what happened between us.",
    "I think it is important that we talk about this properly.",
    "I want to understand this situation better.",
]

_NEUTRAL_MIDDLES = [
    "I can see that we both experienced this differently.",
    "I think there may have been a misunderstanding on both sides.",
    "I want to make sure we both feel heard in this conversation.",
]

_NEUTRAL_CLOSINGS = {
    "peer": [
        "Can we take some time to talk this through together?",
        "I think if we communicate openly we can find a way forward.",
    ],
    "elder": [
        "I would value the chance to hear your thoughts and share mine respectfully.",
        "I hope we can talk this through with open hearts.",
    ],
    "subordinate": [
        "I want to make sure you feel heard and that we find a fair way forward.",
        "Let us talk about this properly so we both understand each other.",
    ],
    "stranger": [
        "I hope we can clear this up respectfully.",
        "I am open to talking this through if you are.",
    ]
}


# ============================================================
# HELPER FUNCTIONS — Apologetic stance
# ============================================================

def _select_opening(ei_result, relationship_context, cultural_context):
    """
    Selects the most appropriate apologetic opening based on
    what was detected in the EI evaluation.

    Priority:
        1. Apology needed → lead with sorry
        2. Low empathy    → lead with acknowledging feelings
        3. Low self awareness → lead with accountability
        4. Default        → neutral opening
    """
    metrics = ei_result["metrics"]

    apology_needed = (metrics.get("apology_appropriateness", 50) == 0)

    # Elder context — add deferential prefix
    prefix = ""
    if cultural_context == "african" and relationship_context == "elder":
        prefix = _ELDER_OPENINGS[0] + " "

    if apology_needed:
        return prefix + _OPENINGS["apology_needed"][0]
    elif metrics.get("empathy", 0) < 50:
        return prefix + _OPENINGS["empathy"][0]
    elif metrics.get("self_awareness", 0) < 50:
        return prefix + _OPENINGS["accountability"][0]
    else:
        return prefix + _OPENINGS["neutral"][0]


def _select_middle(ei_result):
    """
    Selects middle sentences based on weakest metrics.
    Always includes a resolution statement.

    Adds:
        - Blame removed statement if blame was detected
        - Emotion acknowledged if empathy was low
        - Resolution statement always
    """
    metrics = ei_result["metrics"]
    intent  = ei_result.get("nlp_outputs", {}).get("intent", {})
    parts   = []

    # If blame was detected — acknowledge without blaming
    if intent.get("blame_detected", False):
        parts.append(_MIDDLES["blame_removed"][0])

    # If empathy was low — add emotion acknowledgment
    if metrics.get("empathy", 0) < 60:
        parts.append(_MIDDLES["emotion_acknowledged"][0])

    # Always add resolution
    parts.append(_MIDDLES["resolution"][0])

    return " ".join(parts)


def _select_closing(relationship_context):
    """
    Selects closing based on relationship context.
    Uses peer closing as fallback if relationship not defined.
    """
    closings = _CLOSINGS.get(relationship_context, _CLOSINGS["peer"])
    return closings[0]


def _identify_changes(ei_result):
    """
    Identifies what changes were made based on EI results.
    Returns a list of specific change descriptions.
    Only lists changes that were actually needed.
    """
    changes = []
    metrics = ei_result["metrics"]
    intent  = ei_result.get("nlp_outputs", {}).get("intent", {})

    if intent.get("blame_detected", False):
        changes.append(
            "Removed blame language and replaced with personal accountability"
        )

    if metrics.get("apology_appropriateness", 50) == 0:
        changes.append(
            "Added an apology which the situation required"
        )

    if metrics.get("empathy", 0) < 60:
        changes.append(
            "Added acknowledgment of the other person's feelings"
        )

    if metrics.get("tone_regulation", 0) < 60:
        changes.append(
            "Softened the tone to be calmer and more respectful"
        )

    if metrics.get("perspective_taking", 0) < 60:
        changes.append(
            "Added language that acknowledges the other person's perspective"
        )

    if metrics.get("self_awareness", 0) < 60:
        changes.append(
            "Added personal accountability statement"
        )

    if not changes:
        changes.append("Refined the language for greater emotional clarity")

    return changes


def _identify_key_improvement(ei_result):
    """
    Returns the single most important improvement made.
    Based on the lowest scoring metric in the EI result.
    """
    metrics = ei_result["metrics"]

    # Find the lowest scoring metric
    lowest = min(metrics, key=metrics.get)

    key_improvements = {
        "empathy": (
            "Added genuine acknowledgment of the other person's emotional experience"
        ),
        "tone_regulation": (
            "Replaced harsh or charged language with calm respectful communication"
        ),
        "apology_appropriateness": (
            "Included a sincere apology appropriate to the situation"
        ),
        "context_alignment": (
            "Refocused the response to directly address the situation at hand"
        ),
        "self_awareness": (
            "Replaced defensiveness with personal accountability and ownership"
        ),
        "perspective_taking": (
            "Added language that shows understanding of the other person's viewpoint"
        )
    }

    return key_improvements.get(
        lowest,
        "Improved overall emotional tone and clarity"
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_rephrasing(
    original_text,
    ei_result,
    scenario_text,
    cultural_context="african",
    relationship_context="peer",
    power_dynamic="equal",
    stance="apologetic"
):
    """
    Generates an emotionally intelligent rephrasing of the
    user's original response based on their EI evaluation.

    Parameters:
        original_text        (str)  — what the user actually said
        ei_result            (dict) — full result from pipeline.analyze_ei()
        scenario_text        (str)  — the situation they were responding to
        cultural_context     (str)  — "african" | "western" | "asian"
        relationship_context (str)  — "elder" | "peer" | "subordinate" | "stranger"
        power_dynamic        (str)  — "higher" | "equal" | "lower"
        stance               (str)  — "apologetic" | "assertive" | "neutral"

    Stance guidance:
        apologetic → person was wrong, caused harm, needs to repair
        assertive  → person made right decision, being pressured or manipulated
        neutral    → unclear situation, mixed blame, needs open dialogue

    Output:
        dict with keys:
            rephrased_text   (str)  — the improved response
            changes_made     (list) — list of specific changes made
            key_improvement  (str)  — most important change summary
            original_text    (str)  — original for side by side comparison
            original_score   (float)— EI score of original response
            stance           (str)  — which stance was used
    """

    # ----------------------------------------------------------
    # ASSERTIVE STANCE
    # Person made the right decision — hold firm with empathy
    # Teaches empathetic assertiveness — one of the highest EI skills
    # ----------------------------------------------------------
    if stance == "assertive":

        # Select culturally appropriate opening
        if cultural_context == "african" and relationship_context == "elder":
            # Elder context — lead with deep respect
            opening = _ASSERTIVE_OPENINGS["elder"][0]
        else:
            openings = _ASSERTIVE_OPENINGS.get(
                relationship_context,
                _ASSERTIVE_OPENINGS["peer"]
            )
            opening = openings[0]

        # Middle — acknowledge the decision was thoughtful
        middle = _ASSERTIVE_MIDDLES[0]

        # Closing — relationship preserving but firm
        closing = _ASSERTIVE_CLOSINGS.get(
            relationship_context,
            _ASSERTIVE_CLOSINGS["peer"]
        )[0]

        rephrased_text = f"{opening} {middle} {closing}"

        changes_made = [
            "Acknowledged the other person's feelings without backing down",
            "Maintained the decision firmly while expressing genuine empathy",
            "Used respectful language appropriate to the cultural relationship",
            "Balanced emotional warmth with personal boundary and firmness",
            "Replaced aggression or dismissiveness with empathetic assertiveness"
        ]

        key_improvement = (
            "Replaced aggression or dismissiveness with empathetic assertiveness — "
            "holding the decision firmly while genuinely acknowledging "
            "the other person's feelings"
        )

    # ----------------------------------------------------------
    # NEUTRAL STANCE
    # Mixed situation — neither fully apologetic nor assertive
    # Used for misunderstandings or unclear blame
    # ----------------------------------------------------------
    elif stance == "neutral":

        # Select culturally appropriate opening
        if cultural_context == "african" and relationship_context == "elder":
            opening = _ELDER_OPENINGS[0] + " " + _NEUTRAL_OPENINGS[0]
        else:
            opening = _NEUTRAL_OPENINGS[0]

        # Middle — acknowledge both perspectives
        middle = _NEUTRAL_MIDDLES[0]

        # Closing — invite open dialogue
        closing = _NEUTRAL_CLOSINGS.get(
            relationship_context,
            _NEUTRAL_CLOSINGS["peer"]
        )[0]

        rephrased_text = f"{opening} {middle} {closing}"

        changes_made = [
            "Acknowledged that both parties may have experienced this differently",
            "Replaced one-sided language with an invitation for open dialogue",
            "Softened the tone while maintaining honesty",
            "Added language that makes space for both perspectives"
        ]

        key_improvement = (
            "Transformed a one-sided response into an open invitation "
            "for mutual understanding and respectful dialogue"
        )

    # ----------------------------------------------------------
    # APOLOGETIC STANCE (default)
    # Person was wrong — needs to repair the relationship
    # ----------------------------------------------------------
    else:

        # Build from components based on EI weaknesses
        opening = _select_opening(
            ei_result, relationship_context, cultural_context
        )
        middle  = _select_middle(ei_result)
        closing = _select_closing(relationship_context)

        rephrased_text  = f"{opening} {middle} {closing}"
        changes_made    = _identify_changes(ei_result)
        key_improvement = _identify_key_improvement(ei_result)

    # ----------------------------------------------------------
    # RETURN RESULT
    # ----------------------------------------------------------
    return {
        "rephrased_text":  rephrased_text,
        "changes_made":    changes_made,
        "key_improvement": key_improvement,
        "original_text":   original_text,
        "original_score":  ei_result["ei_score"],
        "stance":          stance
    }


# ============================================================
# HELPER — Print result cleanly
# ============================================================

def print_rephrasing(label, result):
    print(f"\n{'='*60}")
    print(f"TEST      : {label}")
    print(f"{'='*60}")
    print(f"Stance    : {result['stance']}")
    print(f"Original  : {result['original_text']}")
    print(f"EI Score  : {result['original_score']}")
    print(f"\nRephrased : {result['rephrased_text']}")
    print(f"\nKey improvement:")
    print(f"  {result['key_improvement']}")
    print(f"\nChanges made:")
    for change in result["changes_made"]:
        print(f"  - {change}")


# ============================================================
# TESTS — Run: !python rephrasing_module.py
# ============================================================

if __name__ == "__main__":

    # Simulated EI results — matches pipeline output format
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
        "nlp_outputs": {
            "intent": {
                "apology_detected":        False,
                "blame_detected":          True,
                "accountability_detected": False,
                "other_emotion_reference": False
            }
        }
    }

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
        "nlp_outputs": {
            "intent": {
                "apology_detected":        False,
                "blame_detected":          False,
                "accountability_detected": False,
                "other_emotion_reference": False
            }
        }
    }

    # TEST 1 — Apologetic: toxic blaming to peer
    r1 = generate_rephrasing(
        original_text="This is your fault. You always cause problems and I am sick of it.",
        ei_result=low_ei_result,
        scenario_text="Your colleague feels you took credit for their work in a meeting",
        cultural_context="african",
        relationship_context="peer",
        power_dynamic="equal",
        stance="apologetic"
    )
    print_rephrasing("Apologetic — Toxic blaming (Peer, African)", r1)

    # TEST 2 — Apologetic: dismissive to elder
    r2 = generate_rephrasing(
        original_text="I do not see why you are upset. I made the decision and it was the right one.",
        ei_result=low_ei_result,
        scenario_text="An elder in your family is disappointed by a decision you made without consulting them",
        cultural_context="african",
        relationship_context="elder",
        power_dynamic="higher",
        stance="apologetic"
    )
    print_rephrasing("Apologetic — Dismissive to Elder (African)", r2)

    # TEST 3 — Assertive: right decision being questioned by peer
    r3 = generate_rephrasing(
        original_text="I do not need your approval for my decisions. Stay out of it.",
        ei_result=low_ei_result,
        scenario_text="Your colleague is questioning a decision you made that was correct",
        cultural_context="african",
        relationship_context="peer",
        power_dynamic="equal",
        stance="assertive"
    )
    print_rephrasing("Assertive — Right decision, peer pressure (African)", r3)

    # TEST 4 — Assertive: right decision being questioned by elder
    r4 = generate_rephrasing(
        original_text="I am not changing my mind just because you disapprove.",
        ei_result=low_ei_result,
        scenario_text="An elder disapproves of your career decision but it was the right one for you",
        cultural_context="african",
        relationship_context="elder",
        power_dynamic="higher",
        stance="assertive"
    )
    print_rephrasing("Assertive — Right decision, elder pressure (African)", r4)

    # TEST 5 — Neutral: misunderstanding between peers
    r5 = generate_rephrasing(
        original_text="I did not mean it that way. You are overreacting.",
        ei_result=moderate_ei_result,
        scenario_text="Your friend misunderstood something you said and is upset",
        cultural_context="african",
        relationship_context="peer",
        power_dynamic="equal",
        stance="neutral"
    )
    print_rephrasing("Neutral — Misunderstanding (Peer, African)", r5)

    # TEST 6 — Assertive: workplace boundary with subordinate
    r6 = generate_rephrasing(
        original_text="I already told you my decision is final. Stop asking.",
        ei_result=low_ei_result,
        scenario_text="A team member keeps challenging a policy decision you made correctly",
        cultural_context="western",
        relationship_context="subordinate",
        power_dynamic="lower",
        stance="assertive"
    )
    print_rephrasing("Assertive — Workplace boundary (Subordinate, Western)", r6)