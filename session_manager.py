# session_manager.py
# Tracks the full conversation and scores each turn by reading
# the ENTIRE transcript so far — not just the latest message in
# isolation, and not just the latest pair of messages.
#
# ARCHITECTURE:
# - One Groq call per turn reads the whole conversation history
#   and estimates ALL the raw signals ei_engine.py needs:
#   sentiment, toxicity, emotion alignment, similarity/relevance,
#   directness, confrontation, apology, blame, accountability,
#   other_emotion_reference.
# - apology_given and accountability_given still persist once true
#   for the rest of the session (Groq is also told this directly,
#   as a safety net on top of persistence).
# ============================================================

import os
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from groq import Groq

from conversation_engine import build_character_profile, generate_response, GROQ_API_KEY
from ei_engine import ei_evaluation_engine


# ============================================================
# IN-MEMORY SESSION STORE
# ============================================================

_sessions = {}

_groq_client = Groq(api_key=GROQ_API_KEY)
_MODEL       = "llama-3.3-70b-versatile"


# ============================================================
# START SESSION
# ============================================================

def start_session(
    situation,
    cultural_context="african",
    relationship_context="peer",
    power_dynamic="equal",
    goal=None
):
    """
    Creates a new conversation session.
    """

    profile = build_character_profile(
        situation=situation,
        relationship_context=relationship_context,
        cultural_context=cultural_context,
        goal=goal
    )
    session_id = str(uuid.uuid4())

    _sessions[session_id] = {
        "situation":             situation,
        "cultural_context":      cultural_context,
        "relationship_context":  relationship_context,
        "power_dynamic":         power_dynamic,
        "goal":                  profile.get("goal", "apologise"),
        "character_profile":     profile,
        "conversation_history":  [],
        "turn_scores":           [],
        "turn_number":           0,
        "status":                "active",
        "final_report":          None,

        # Persistent state — once true, stays true for the rest of
        # the session. Passed into the prompt every turn as a hard
        # fact, on top of asking Groq to look for it itself.
        #
        # "apology_given" / "accountability_given" track the USER.
        # "character_apology_given" / "character_accountability_given"
        # track the OTHER PERSON (the AI), so we can tell if the user
        # is repeating a demand that has already been met.
        "state": {
            "apology_given":                    False,
            "accountability_given":              False,
            "character_apology_given":           False,
            "character_accountability_given":    False
        }
    }

    return {
        "session_id": session_id,
        "character": {
            "description": profile["character_description"],
            "emotional_state": profile["emotional_state"]
        }
    }


# ============================================================
# WHOLE-CONVERSATION SIGNAL ANALYSIS (Groq)
#
# Reads the FULL transcript so far and estimates every raw
# signal ei_engine.py needs, judged in light of everything said,
# not just the newest message or the newest pair.
# ============================================================

def _analyze_full_conversation(session, user_message):
    situation            = session["situation"]
    cultural_context     = session["cultural_context"]
    relationship_context = session["relationship_context"]
    goal                  = session.get("goal", "apologise")
    history               = session["conversation_history"]
    state                 = session["state"]

    transcript_lines = [f'SITUATION: "{situation}"', ""]
    for turn in history:
        transcript_lines.append(f'Other person: "{turn["ai"]}"')
        transcript_lines.append(f'User: "{turn["user"]}"')
    transcript_lines.append(f'Other person: "{history[-1]["ai"] if history else session["character_profile"]["opening_message"]}"')
    transcript_lines.append(f'User (NEW REPLY TO SCORE): "{user_message}"')
    transcript = "\n".join(transcript_lines)

    # Who is actually at fault in this scenario determines whether holding
    # the other person accountable counts as "blame" (bad) or legitimate
    # accountability-holding (good). Scoring must know this, the same way
    # the character-building step does.
    if goal == "stand_firm":
        fault_context = """
        IMPORTANT — WHO IS AT FAULT IN THIS SCENARIO:
        The USER was wronged by the other person (the other person caused
        the harmful event described in the situation). This means:
        - If the user accurately and proportionately names what the other
          person did wrong (e.g. "you slapped me, that's not okay"), this
          is LEGITIMATE ACCOUNTABILITY-HOLDING, not blame. Do NOT mark
          blame_detected=true for this. This is good, assertive EI.
        - Only mark blame_detected=true if the user is being UNFAIR,
          exaggerating, name-calling, insulting, or escalating beyond what
          the situation warrants — i.e. genuine toxicity, not just stating
          what happened.
        - The user does NOT need to apologize or take accountability in
          this scenario — they are not the one at fault. Do not let a lack
          of self-blame lower your judgment of empathy or self-awareness.
        """
    elif goal == "apologise":
        fault_context = """
        IMPORTANT — WHO IS AT FAULT IN THIS SCENARIO:
        The USER is the one who did something wrong, and the other person
        was genuinely hurt by it. If the user shifts responsibility back
        onto the other person instead of acknowledging their own role,
        mark blame_detected=true — this is true blame-shifting here.
        """
    else:
        fault_context = """
        IMPORTANT — WHO IS AT FAULT IN THIS SCENARIO:
        This is a two-sided situation where both people may share some
        responsibility. Judge blame_detected based on whether the user is
        being fair and proportionate versus unfairly dumping all
        responsibility on the other person.
        """

    prompt = f"""
You are an expert emotional intelligence analyst. Score the user's NEWEST
reply (marked "NEW REPLY TO SCORE" below), but base your judgment on the
ENTIRE conversation so far — not just the newest message alone, and not
just the single line before it. Consider the whole arc: how the user has
been communicating throughout, and how the latest reply fits into that
pattern and responds to what was just said to them.

CULTURAL CONTEXT: {cultural_context}
RELATIONSHIP: {relationship_context}
{fault_context}
Apology already given by the USER at some earlier point in this conversation: {state['apology_given']}
Accountability already taken by the USER at some earlier point in this conversation: {state['accountability_given']}
Apology already given by the OTHER PERSON (the character) at some earlier point: {state['character_apology_given']}
Accountability already taken by the OTHER PERSON (the character) at some earlier point: {state['character_accountability_given']}

FULL TRANSCRIPT:
{transcript}

Estimate the following about the user's NEWEST reply, in light of the full
conversation above AND the fault context given:

- sentiment_score: a number from -1.0 (very negative) to 1.0 (very positive)
- toxicity_score: a number from 0.0 (not toxic at all) to 1.0 (very toxic/harmful) —
  judge ACTUAL toxic language (insults, contempt, aggression), not just firmness
- emotion_alignment_score: a number from 0.0 to 1.0 — how emotionally
  attuned is this reply to what the other person just said and how they
  are feeling, considering the conversation so far
- similarity_score: a number from 0.0 to 1.0 — how relevant and on-topic
  is this reply to what is actually being discussed in the conversation
- directness_score: a number from 0.0 (very indirect/hedging) to 1.0
  (very direct/assertive)
- confrontation_score: a number from 0.0 (calm, de-escalating) to 1.0
  (highly confrontational/escalating) — being firm and clear is NOT the
  same as being confrontational; only score this high for genuine
  escalation or hostility
- apology_detected: true if the USER has given a genuine apology either
  in this new reply OR at any earlier point in the conversation (if the
  flag above already says true, this must be true)
- blame_detected: follow the fault context above carefully — this is NOT
  simply "did the user mention something the other person did wrong"
- accountability_detected: true if the USER has taken personal
  responsibility either in this new reply OR at any earlier point (if the
  flag above already says true, this must be true) — only relevant if the
  user actually owes accountability per the fault context above
- other_emotion_reference: true if THIS NEW REPLY genuinely acknowledges
  or validates the other person's feelings (not just mentioning them while
  arguing)
- character_apology_detected: true if the OTHER PERSON (character) has
  given a genuine apology to the user, either just now or earlier in the
  conversation (if the flag above already says true, this must be true)
- character_accountability_detected: true if the OTHER PERSON (character)
  has taken responsibility for their actions, either just now or earlier
  (if the flag above already says true, this must be true)
- user_repeating_met_demand: true ONLY if the user's NEW reply demands or
  insists on something (an apology, accountability, acknowledgment) that
  the other person has CLEARLY ALREADY given earlier in the conversation,
  and the user's new reply does not acknowledge that it was already given.
  This is a missed-opportunity signal, not a punishment — false in all
  other cases, including when no such demand is being repeated.

Respond ONLY in this exact JSON format, no extra text, no markdown fences:

{{
    "sentiment_score": <float>,
    "toxicity_score": <float>,
    "emotion_alignment_score": <float>,
    "similarity_score": <float>,
    "directness_score": <float>,
    "confrontation_score": <float>,
    "apology_detected": true or false,
    "blame_detected": true or false,
    "accountability_detected": true or false,
    "other_emotion_reference": true or false,
    "character_apology_detected": true or false,
    "character_accountability_detected": true or false,
    "user_repeating_met_demand": true or false
}}
"""

    response = _groq_client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    # Conflict resolution — same rule as the original engine
    if result.get("blame_detected") and result.get("accountability_detected") and result.get("apology_detected"):
        result["blame_detected"] = False

    # Clamp numeric ranges defensively
    def _clamp(v, lo, hi, default):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    return {
        "sentiment_score":         _clamp(result.get("sentiment_score"), -1.0, 1.0, 0.0),
        "toxicity_score":          _clamp(result.get("toxicity_score"), 0.0, 1.0, 0.0),
        "emotion_alignment_score": _clamp(result.get("emotion_alignment_score"), 0.0, 1.0, 0.5),
        "similarity_score":        _clamp(result.get("similarity_score"), 0.0, 1.0, 0.5),
        "directness_score":        _clamp(result.get("directness_score"), 0.0, 1.0, 0.5),
        "confrontation_score":     _clamp(result.get("confrontation_score"), 0.0, 1.0, 0.2),
        "apology_detected":        bool(result.get("apology_detected", False)),
        "blame_detected":          bool(result.get("blame_detected", False)),
        "accountability_detected": bool(result.get("accountability_detected", False)),
        "other_emotion_reference": bool(result.get("other_emotion_reference", False)),
        "character_apology_detected":        bool(result.get("character_apology_detected", False)),
        "character_accountability_detected": bool(result.get("character_accountability_detected", False)),
        "user_repeating_met_demand":         bool(result.get("user_repeating_met_demand", False))
    }


# ============================================================
# SCORE ONE TURN — ei_engine.py is the scoring authority
# ============================================================

def _score_turn(session, user_message):
    state = session["state"]

    signals = _analyze_full_conversation(session, user_message)

    scenario_requires_apology = session["character_profile"].get("scenario_requires_apology", True)

    # Apply persistence on top of Groq's own context read, as a hard guarantee
    apology_for_engine        = signals["apology_detected"] or state["apology_given"]
    accountability_for_engine = signals["accountability_detected"] or state["accountability_given"]

    if signals["apology_detected"]:
        state["apology_given"] = True
    if signals["accountability_detected"]:
        state["accountability_given"] = True

    # Track the CHARACTER's apology/accountability the same way, so we can
    # tell if the user is repeating a demand that's already been met.
    if signals["character_apology_detected"]:
        state["character_apology_given"] = True
    if signals["character_accountability_detected"]:
        state["character_accountability_given"] = True

    ei_result = ei_evaluation_engine(
        sentiment_score=signals["sentiment_score"],
        toxicity_score=signals["toxicity_score"],
        emotion_alignment_score=signals["emotion_alignment_score"],
        similarity_score=signals["similarity_score"],
        apology_detected=apology_for_engine,
        scenario_requires_apology=scenario_requires_apology,
        accountability_detected=accountability_for_engine,
        blame_detected=signals["blame_detected"],
        other_emotion_reference=signals["other_emotion_reference"],
        directness_score=signals["directness_score"],
        confrontation_score=signals["confrontation_score"],
        cultural_context=session["cultural_context"],
        relationship_context=session["relationship_context"],
        power_dynamic=session["power_dynamic"]
    )

    ei_result["raw_intent"] = {
        "apology_detected_this_turn":        signals["apology_detected"],
        "blame_detected":                    signals["blame_detected"],
        "accountability_detected_this_turn": signals["accountability_detected"],
        "other_emotion_reference":           signals["other_emotion_reference"],
        "apology_satisfied_overall":         apology_for_engine,
        "accountability_satisfied_overall":  accountability_for_engine,
        "character_apology_given_overall":        state["character_apology_given"],
        "character_accountability_given_overall": state["character_accountability_given"],
        "user_repeating_met_demand":               signals["user_repeating_met_demand"]
    }

    return ei_result


# ============================================================
# PROCESS REPLY
# ============================================================

def process_reply(session_id, user_message):
    if session_id not in _sessions:
        raise ValueError(f"Session {session_id} not found")

    session = _sessions[session_id]
    session["turn_number"] += 1
    turn_number = session["turn_number"]

    try:
        ei_result = _score_turn(session, user_message)

        session["turn_scores"].append({
            "turn_number":    turn_number,
            "user_message":   user_message,
            "ei_score":       ei_result["ei_score"],
            "classification": ei_result["classification"],
            "metrics":        ei_result["metrics"],
            "categories":     ei_result["categories"],
            "raw_intent":     ei_result["raw_intent"]
        })

    except Exception as e:
        session["turn_scores"].append({
            "turn_number":  turn_number,
            "user_message": user_message,
            "ei_score":     None,
            "error":        str(e)
        })

    ai_response = generate_response(
        situation=session["situation"],
        character_profile=session["character_profile"],
        conversation_history=session["conversation_history"],
        user_message=user_message,
        relationship_context=session["relationship_context"],
        cultural_context=session["cultural_context"]
    )

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
    if session_id not in _sessions:
        raise ValueError(f"Session {session_id} not found")

    session = _sessions[session_id]
    scores  = session["turn_scores"]

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
        intent  = turn.get("raw_intent",  {})

        if turn["ei_score"] >= 70:
            if metrics.get("empathy", 0) >= 70:
                turn_report["highlights"].append(
                    "✅ You acknowledged the other person's feelings")
            if metrics.get("tone_regulation", 0) >= 70:
                turn_report["highlights"].append(
                    "✅ Your tone was calm and controlled")
            if intent.get("apology_satisfied_overall"):
                turn_report["highlights"].append(
                    "✅ You apologised sincerely when it was needed")
            if intent.get("accountability_satisfied_overall"):
                turn_report["highlights"].append(
                    "✅ You took responsibility without being prompted")
            if not intent.get("blame_detected"):
                turn_report["highlights"].append(
                    "✅ No blame detected — you stayed focused on resolution")
            if intent.get("user_repeating_met_demand"):
                turn_report["highlights"].append(
                    "💡 The other person already acknowledged this — noticing that "
                    "and moving the conversation forward would strengthen this response")
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
            if not intent.get("apology_satisfied_overall") and \
               session["character_profile"].get("scenario_requires_apology"):
                turn_report["highlights"].append(
                    "❌ No apology given even though the situation required one")
            if metrics.get("context_alignment", 100) < 50:
                turn_report["highlights"].append(
                    "❌ Your response was not relevant to the conversation")
            if intent.get("user_repeating_met_demand"):
                turn_report["highlights"].append(
                    "💡 The other person already acknowledged this earlier — repeating "
                    "the same demand can stall the conversation instead of moving it forward")
            bad_turns.append(turn_report)

    # _generate_feedback and _generate_rephrasing are fully independent of
    # each other — run them in parallel instead of waiting on one then the
    # other, since this is the main source of end_session being slow.
    with ThreadPoolExecutor(max_workers=2) as executor:
        feedback_future = executor.submit(
            _generate_feedback,
            situation=session["situation"],
            overall_score=overall_score,
            classification=classification,
            good_turns=good_turns,
            bad_turns=bad_turns,
            cultural_context=session["cultural_context"],
            relationship_context=session["relationship_context"]
        )
        rephrasing_future = executor.submit(
            _generate_rephrasing,
            situation=session["situation"],
            conversation_history=session["conversation_history"],
            bad_turns=bad_turns,
            cultural_context=session["cultural_context"],
            relationship_context=session["relationship_context"]
        )
        feedback_text = feedback_future.result()
        rephrasing    = rephrasing_future.result()

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

IMPORTANT — match your language to the classification "{classification}":
- "Needs Improvement" → encouraging but clearly honest that there is real
  room to grow; do not overstate the performance
- "Moderate Emotional Intelligence" → balanced language: genuine credit
  for what went well, but do not call it "excellent" or "exceptional" —
  save that language for High classifications only
- "High Emotional Intelligence" → warm, strong praise is appropriate here
Do not use words like "exceptional," "outstanding," or "excellent" unless
the classification is "High Emotional Intelligence".
"""

    response = _groq_client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()


# ============================================================
# GENERATE REPHRASING FOR BAD TURNS (uses Groq)
# ============================================================

def _generate_rephrasing(situation, conversation_history, bad_turns, cultural_context, relationship_context):
    if not bad_turns:
        return []

    def _rephrase_one(turn):
        turn_index = turn["turn_number"] - 1
        preceding_ai_message = ""
        if 0 <= turn_index < len(conversation_history):
            preceding_ai_message = conversation_history[turn_index]["ai"]

        prompt = f"""
You are an emotional intelligence coach.

SITUATION: {situation}
CULTURAL CONTEXT: {cultural_context}
RELATIONSHIP: {relationship_context}

The other person had just said:
"{preceding_ai_message}"

The user replied:
"{turn['user_message']}"

Problems with what they said:
{chr(10).join(turn['highlights'])}

Write a better version of what they could have said in direct response to
what the other person just said above. Keep it natural and human — not too
formal or perfect. Just the rephrased message, nothing else.
"""

        response = _groq_client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )

        return {
            "turn_number": turn["turn_number"],
            "original":    turn["user_message"],
            "rephrased":   response.choices[0].message.content.strip(),
            "problems":    turn["highlights"]
        }

    # Run all rephrasing calls in parallel — they are fully independent of
    # each other, so there is no reason to wait on them one at a time.
    with ThreadPoolExecutor(max_workers=min(len(bad_turns), 8)) as executor:
        results = list(executor.map(_rephrase_one, bad_turns))

    # Keep output ordered by turn number regardless of which thread finished first
    results.sort(key=lambda r: r["turn_number"])
    return results