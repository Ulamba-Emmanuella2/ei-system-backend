# session_manager.py
# Tracks the full conversation, scores each turn silently,
# and builds the final EI report at the end
# ============================================================

import uuid
from pipeline_cloud import analyze_ei_cloud as analyze_ei
from conversation_engine import build_character_profile, generate_response, GROQ_API_KEY


# ============================================================
# IN-MEMORY SESSION STORE
# ============================================================

_sessions = {}


# ============================================================
# START SESSION
# ============================================================

def start_session(
    situation,
    cultural_context="african",
    relationship_context="peer",
    power_dynamic="equal"
):
    """
    Creates a new conversation session.

    Input:
        situation            (str) — what the user described
        cultural_context     (str) — african / western / asian
        relationship_context (str) — elder / peer / subordinate / stranger
        power_dynamic        (str) — higher / equal / lower

    Output:
        dict with:
            session_id      (str)  — unique ID for this session
            opening_message (str)  — first message from the AI character
            character       (dict) — the character profile
    """

    profile    = build_character_profile(
        situation=situation,
        relationship_context=relationship_context,
        cultural_context=cultural_context
    )
    session_id = str(uuid.uuid4())

    _sessions[session_id] = {
        "situation":            situation,
        "cultural_context":     cultural_context,
        "relationship_context": relationship_context,
        "power_dynamic":        power_dynamic,
        "character_profile":    profile,
        "conversation_history": [],
        "turn_scores":          [],
        "turn_number":          0,
        "status":               "active",
        "final_report":         None
    }

    return {
        "session_id":      session_id,
        "opening_message": profile["opening_message"],
        "character": {
            "description":     profile["character_description"],
            "emotional_state": profile["emotional_state"]
        }
    }


# ============================================================
# PROCESS REPLY
# ============================================================

def process_reply(session_id, user_message):
    """
    Processes one user reply:
    1. Scores the reply silently using the EI pipeline
    2. Generates the AI character's response
    3. Stores everything in the session

    Input:
        session_id   (str) — from start_session
        user_message (str) — what the user just typed

    Output:
        dict with:
            ai_response  (str) — the character's next message
            turn_number  (int) — which turn this was
    """

    if session_id not in _sessions:
        raise ValueError(f"Session {session_id} not found")

    session = _sessions[session_id]
    session["turn_number"] += 1
    turn_number = session["turn_number"]

    # ----------------------------------------------------------
    # SCORE THIS TURN SILENTLY
    # User never sees this — stored for the final report
    # ----------------------------------------------------------
    try:
        ei_result = analyze_ei(
            scenario_text=session["situation"],
            response_text=user_message,
            scenario_requires_apology=session["character_profile"]["scenario_requires_apology"],
            cultural_context=session["cultural_context"],
            relationship_context=session["relationship_context"],
            power_dynamic=session["power_dynamic"]
        )

        session["turn_scores"].append({
            "turn_number":    turn_number,
            "user_message":   user_message,
            "ei_score":       ei_result["ei_score"],
            "classification": ei_result["classification"],
            "metrics":        ei_result["metrics"],
            "categories":     ei_result["categories"],
            "nlp_outputs":    ei_result.get("nlp_outputs", {})
        })

    except Exception as e:
        session["turn_scores"].append({
            "turn_number":  turn_number,
            "user_message": user_message,
            "ei_score":     None,
            "error":        str(e)
        })

    # ----------------------------------------------------------
    # GENERATE AI RESPONSE
    # ----------------------------------------------------------
    ai_response = generate_response(
        situation=session["situation"],
        character_profile=session["character_profile"],
        conversation_history=session["conversation_history"],
        user_message=user_message,
        relationship_context=session["relationship_context"],
        cultural_context=session["cultural_context"]
    )

    # ----------------------------------------------------------
    # STORE THIS TURN IN HISTORY
    # ----------------------------------------------------------
    session["conversation_history"].append({
        "user": user_message,
        "ai":   ai_response
    })

    return {
        "ai_response": ai_response,
        "turn_number": turn_number
    }


# ============================================================
# END SESSION — BUILD FINAL REPORT
# ============================================================

def end_session(session_id):
    """
    Ends the conversation and builds the full EI report.
    Session is kept in memory so /report can re-fetch it.

    Input:
        session_id (str) — from start_session

    Output:
        Full EI report dict
    """

    if session_id not in _sessions:
        raise ValueError(f"Session {session_id} not found")

    session = _sessions[session_id]
    scores  = session["turn_scores"]

    # ----------------------------------------------------------
    # CALCULATE OVERALL SCORE
    # ----------------------------------------------------------
    valid_scores = [t["ei_score"] for t in scores if t["ei_score"] is not None]

    if valid_scores:
        overall_score = round(sum(valid_scores) / len(valid_scores), 1)
    else:
        overall_score = 0.0

    if overall_score >= 80:
        classification = "High Emotional Intelligence"
    elif overall_score >= 50:
        classification = "Moderate Emotional Intelligence"
    else:
        classification = "Needs Improvement"

    # ----------------------------------------------------------
    # IDENTIFY GOOD AND BAD TURNS
    # ----------------------------------------------------------
    good_turns = []
    bad_turns  = []

    for turn in scores:
        if turn["ei_score"] is None:
            continue

        turn_report = {
            "turn_number":    turn["turn_number"],
            "user_message":   turn["user_message"],
            "ei_score":       turn["ei_score"],
            "classification": turn["classification"],
            "highlights":     []
        }

        metrics = turn.get("metrics",     {})
        nlp     = turn.get("nlp_outputs", {})
        intent  = nlp.get("intent",       {})

        if turn["ei_score"] >= 70:
            if metrics.get("empathy", 0) >= 70:
                turn_report["highlights"].append(
                    "✅ You acknowledged the other person's feelings")
            if metrics.get("tone_regulation", 0) >= 70:
                turn_report["highlights"].append(
                    "✅ Your tone was calm and controlled")
            if intent.get("apology_detected"):
                turn_report["highlights"].append(
                    "✅ You apologised sincerely when it was needed")
            if intent.get("accountability_detected"):
                turn_report["highlights"].append(
                    "✅ You took responsibility without being prompted")
            if not intent.get("blame_detected"):
                turn_report["highlights"].append(
                    "✅ No blame detected — you stayed focused on resolution")
            good_turns.append(turn_report)

        else:
            if metrics.get("empathy", 100) < 50:
                turn_report["highlights"].append(
                    "❌ Low empathy — you did not acknowledge the other person's feelings")
            if metrics.get("tone_regulation", 100) < 50:
                turn_report["highlights"].append(
                    "❌ Poor tone — your language was aggressive or harmful")
            if intent.get("blame_detected"):
                turn_report["highlights"].append(
                    "❌ Blame detected — you shifted responsibility to the other person")
            if not intent.get("apology_detected") and \
               session["character_profile"]["scenario_requires_apology"]:
                turn_report["highlights"].append(
                    "❌ No apology given even though the situation required one")
            if metrics.get("context_alignment", 100) < 50:
                turn_report["highlights"].append(
                    "❌ Your response was not relevant to the situation")
            bad_turns.append(turn_report)

    # ----------------------------------------------------------
    # GENERATE WRITTEN FEEDBACK AND REPHRASING
    # ----------------------------------------------------------
    feedback_text = _generate_feedback(
        situation=session["situation"],
        overall_score=overall_score,
        classification=classification,
        good_turns=good_turns,
        bad_turns=bad_turns,
        cultural_context=session["cultural_context"],
        relationship_context=session["relationship_context"]
    )

    rephrasing = _generate_rephrasing(
        situation=session["situation"],
        bad_turns=bad_turns,
        cultural_context=session["cultural_context"],
        relationship_context=session["relationship_context"]
    )

    # ----------------------------------------------------------
    # BUILD FINAL REPORT
    # ----------------------------------------------------------
    conversation_history = session["conversation_history"]

    final_report = {
        "overall_ei_score":     overall_score,
        "classification":       classification,
        "what_you_did_right":   good_turns,
        "where_you_went_wrong": bad_turns,
        "feedback":             feedback_text,
        "rephrasing":           rephrasing,
        "conversation_history": conversation_history,
        "total_turns":          len(scores)
    }

    # Keep session in memory — do NOT delete
    # /report endpoint can re-fetch this any time
    session["status"]       = "ended"
    session["final_report"] = final_report

    return final_report


# ============================================================
# GET SESSION (used by api.py)
# ============================================================

def get_session(session_id):
    return _sessions.get(session_id)


# ============================================================
# GENERATE WRITTEN FEEDBACK (uses Groq)
# ============================================================

def _generate_feedback(
    situation, overall_score, classification,
    good_turns, bad_turns,
    cultural_context, relationship_context
):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    good_summary = "\n".join([
        f"Turn {t['turn_number']}: {', '.join(t['highlights'])}"
        for t in good_turns
    ]) or "None"

    bad_summary = "\n".join([
        f"Turn {t['turn_number']}: {', '.join(t['highlights'])}"
        for t in bad_turns
    ]) or "None"

    prompt = f"""
You are an emotional intelligence coach giving feedback after a conversation training session.

SITUATION: {situation}
CULTURAL CONTEXT: {cultural_context}
RELATIONSHIP: {relationship_context}
OVERALL EI SCORE: {overall_score} — {classification}

WHAT THEY DID RIGHT:
{good_summary}

WHERE THEY WENT WRONG:
{bad_summary}

Write 3 to 4 sentences of warm, constructive, specific feedback.
Tell them what their biggest strength was, what their biggest weakness was,
and one specific thing they should focus on improving.
Speak directly to them using "you". Be encouraging but honest.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()


# ============================================================
# GENERATE REPHRASING FOR BAD TURNS (uses Groq)
# ============================================================

def _generate_rephrasing(situation, bad_turns, cultural_context, relationship_context):
    if not bad_turns:
        return []

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    rephrased = []

    for turn in bad_turns:
        prompt = f"""
You are an emotional intelligence coach.

SITUATION: {situation}
CULTURAL CONTEXT: {cultural_context}
RELATIONSHIP: {relationship_context}

The person said this during the conversation:
"{turn['user_message']}"

Problems with what they said:
{chr(10).join(turn['highlights'])}

Write a better version of what they could have said.
Keep it natural and human — not too formal or perfect.
Just the rephrased message, nothing else.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )

        rephrased.append({
            "turn_number": turn["turn_number"],
            "original":    turn["user_message"],
            "rephrased":   response.choices[0].message.content.strip(),
            "problems":    turn["highlights"]
        })

    return rephrased