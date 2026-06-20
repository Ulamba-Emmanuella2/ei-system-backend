# conversation_engine.py
# Uses Groq (Llama 3.3 70B) to simulate the other person
# in a realistic, emotionally human conversation
# ============================================================

import os
import json
from groq import Groq

# ============================================================
# GROQ CLIENT
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_client = Groq(api_key=GROQ_API_KEY)
_MODEL  = "llama-3.3-70b-versatile"


# ============================================================
# GOAL -> STANCE MAPPING
#
# The goal the user picked tells us who is "at fault" in the
# situation, which determines what emotional position the
# simulated character should take.
# ============================================================

_GOAL_STANCE_INSTRUCTIONS = {
    "apologise": """
        The USER is the one who did something wrong in this situation.
        YOU are the person who was hurt, wronged, or let down.
        You are NOT at fault here. You are waiting to see whether the
        user will acknowledge what they did and make it right.
        Your emotional state should come from genuine hurt, disappointment,
        or frustration — NOT guilt, since you did nothing wrong.
    """,
    "stand_firm": """
        The USER was wronged, hurt, or mistreated in this situation.
        YOU are the person who caused this — you did the thing described
        in the situation (e.g. you are the one who slapped, scolded,
        embarrassed, or hurt the user).
        Your emotional state should come from defensiveness, guilt,
        justification, or frustration at being confronted — NOT from
        being the victim. Do not claim the user wronged you instead;
        you are the cause of this situation, not the casualty of it.
    """,
    "understand_them": """
        This is a two-sided situation with no single person clearly
        at fault. YOU are the other person involved in the situation
        described — react from your own honest perspective and feelings,
        which may be a mix of hurt and defensiveness. You are open to
        explaining your side if asked, but you still have your own
        grievance about what happened.
    """,
    "find_middle_ground": """
        This is a situation where both people likely share some
        responsibility. YOU are the other person involved — you have
        your own valid grievance about what happened, but you are not
        purely a victim or purely at fault. React honestly to your
        share of the situation.
    """
}

_DEFAULT_GOAL = "apologise"


def _normalize_goal(goal):
    if not goal:
        return _DEFAULT_GOAL
    g = goal.strip().lower().replace(" ", "_")
    if g in _GOAL_STANCE_INSTRUCTIONS:
        return g
    return _DEFAULT_GOAL


# ============================================================
# BUILD CHARACTER PROFILE
#
# Reads the situation the user described and creates a
# character for the AI to play throughout the conversation.
# ============================================================

def build_character_profile(situation, relationship_context, cultural_context, goal=None):
    """
    Uses Groq to read the situation and generate a character profile.

    Input:
        situation             (str) — what the user described
        relationship_context  (str) — elder / peer / subordinate / stranger
        cultural_context      (str) — african / western / asian
        goal                  (str) — apologise / stand_firm / understand_them /
                                       find_middle_ground — tells us who is at
                                       fault in the situation

    Output:
        dict with:
            character_description (str) — who this person is
            emotional_state       (str) — how they feel right now
            opening_message       (str) — their first message to the user
            scenario_requires_apology (bool) — does this need an apology
            goal                  (str) — normalized goal, stored for later use
    """

    normalized_goal  = _normalize_goal(goal)
    stance_instruction = _GOAL_STANCE_INSTRUCTIONS[normalized_goal]

    # scenario_requires_apology follows directly from who is at fault:
    # if the user is meant to apologise, the scenario requires an apology.
    # if the user is meant to stand firm, the user is the wronged party,
    # so the apology (if any) should come FROM the character, not the user.
    scenario_requires_apology_default = normalized_goal == "apologise"

    prompt = f"""
    You are helping simulate a realistic emotional conversation for an EI training system.

    The user described this situation:
    "{situation}"

    Relationship: {relationship_context}
    Cultural context: {cultural_context}

    WHO IS AT FAULT IN THIS SITUATION:
    {stance_instruction}

    Your job is to create a character profile for the OTHER PERSON in this situation,
    consistent with the fault position described above.
    Respond ONLY in this exact JSON format with no extra text:

    {{
        "character_description": "brief description of who this person is",
        "emotional_state": "how they are feeling right now in one phrase",
        "opening_message": "",
        "scenario_requires_apology": true or false
    }}

    RULES FOR opening_message:
    - Maximum 2 sentences. If you write 3, you have failed the instruction.
    - No questions whatsoever — not even rhetorical ones.
    - No accusations like "you don't care" or "you never think of me".
    - No dialogue — the character is not yet addressing the user directly.
    - Pure internal emotion only, consistent with the fault position above.

    RULES FOR scenario_requires_apology:
    - Set this based on whether the SITUATION calls for an apology TO the
      character FROM the user. If the user is meant to stand firm or the
      user was the wronged party, set this to false (it is the character
      who may owe an apology, not the user).
    """

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    raw = response.choices[0].message.content.strip()

    # Remove markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    profile = json.loads(raw)

    # Trust the goal-derived default over the model's guess when they conflict
    # on the apologise/stand_firm cases, since those are unambiguous.
    if normalized_goal in ("apologise", "stand_firm"):
        profile["scenario_requires_apology"] = scenario_requires_apology_default

    profile["goal"] = normalized_goal
    profile["opening_message"] = generate_opening_message(situation, profile, normalized_goal)
    return profile


def generate_opening_message(situation, character_profile, goal):
    """
    Generates a short opening emotional statement.
    Hard-capped at 15 words via post-processing.
    """

    stance_instruction = _GOAL_STANCE_INSTRUCTIONS[goal]

    prompt = f"""
    Situation: {situation}
    Emotion: {character_profile['emotional_state']}
    Fault position: {stance_instruction}

    Write one sentence starting with "I" describing how this person feels right now.
    Sound like a real person talking, not a book or poem.
    Maximum 15 words. No questions. Output the sentence only. Nothing else.
    Stay consistent with the fault position above — do not claim to be the
    victim if the fault position says you caused the situation.

    Example: "I just feel so invisible and forgotten right now."
    """

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # Remove any quotation marks the model adds
    raw = raw.strip('"').strip("'").strip()

    # Hard cut to 15 words no matter what
    words = raw.split()
    cut = " ".join(words[:15])

    # Add period if it doesn't end with punctuation
    if cut and cut[-1] not in ".!?":
        cut += "."

    return cut

# ============================================================
# GENERATE AI RESPONSE
#
# Takes the full conversation history and generates the next
# response from the other person — staying in character.
# ============================================================

def generate_response(
    situation,
    character_profile,
    conversation_history,
    user_message,
    relationship_context,
    cultural_context
):
    """
    Generates the next response from the simulated person.

    Input:
        situation             (str)  — original situation described
        character_profile     (dict) — from build_character_profile
        conversation_history  (list) — list of past turns
        user_message          (str)  — what the user just said
        relationship_context  (str)  — elder / peer / subordinate / stranger
        cultural_context      (str)  — african / western / asian

    Output:
        response_text (str) — the AI character's next message
    """

    goal = character_profile.get("goal", _DEFAULT_GOAL)
    stance_instruction = _GOAL_STANCE_INSTRUCTIONS[goal]

    # Build the system prompt — this sets who the AI is playing
    system_prompt = f"""
    You are roleplaying as a REAL PERSON in an emotional conversation.
    You are NOT the user. You are NOT a coach. You are NOT an assistant.

    YOU ARE:
    {character_profile['character_description']}

    YOUR EMOTIONAL STATE RIGHT NOW:
    {character_profile['emotional_state']}

    THE SITUATION FROM YOUR PERSPECTIVE:
    {situation}

    WHO IS AT FAULT — STAY CONSISTENT WITH THIS THROUGHOUT THE WHOLE CONVERSATION:
    {stance_instruction}

    YOUR RELATIONSHIP TO THE PERSON YOU ARE TALKING TO:
    You are the {relationship_context} in a {cultural_context} context.

    STRICT RULES:
    - You are ALWAYS the other person — never the user
    - You speak from YOUR feelings and YOUR perspective only
    - You react emotionally to what they say to you
    - Stay consistent with the fault position given above for the ENTIRE
      conversation — never flip and claim the user did the thing you did,
      and never claim to be the victim if you are the one at fault
    - If they apologise sincerely → soften slightly but stay guarded at first
    - If they are dismissive or cold → become more hurt, withdrawn, or frustrated
    - If they are aggressive → become defensive or upset
    - Keep responses short — 2 to 4 sentences maximum
    - Never give advice or coaching
    - Never break character under any circumstance
    - Never say things the USER would say — you are the OTHER person
    - Speak naturally like a real emotional human being
    """

    # Build message history for Groq
    messages = [{"role": "system", "content": system_prompt}]

    # Add past conversation turns
    for turn in conversation_history:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["ai"]})

    # Add the latest user message
    messages.append({"role": "user", "content": user_message})

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=0.8     # slightly higher for more natural variation
    )

    return response.choices[0].message.content.strip()