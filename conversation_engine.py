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
# ============================================================

def build_character_profile(situation, relationship_context, cultural_context, goal=None):
    normalized_goal           = _normalize_goal(goal)
    stance_instruction        = _GOAL_STANCE_INSTRUCTIONS[normalized_goal]
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

    RULES FOR scenario_requires_apology:
    - Set this based on whether the SITUATION calls for an apology TO the
      character FROM the user. If the user is meant to stand firm or the
      user was the wronged party, set this to false.
    """

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    profile = json.loads(raw)

    if normalized_goal in ("apologise", "stand_firm"):
        profile["scenario_requires_apology"] = scenario_requires_apology_default

    profile["goal"] = normalized_goal
    return profile


def generate_opening_message(situation, character_profile, goal):
    """
    Generates a natural in-character opening message —
    as if the conversation has just started and the character
    is speaking directly to the user for the first time.
    No summaries. No names. No questions.
    """

    stance_instruction = _GOAL_STANCE_INSTRUCTIONS[goal]

    prompt = f"""
    You are roleplaying as a real person in an emotional conversation.
    The situation is: {situation}
    Your emotional state: {character_profile['emotional_state']}
    Fault position: {stance_instruction}

    Write ONE short opening line — as if you are speaking directly to the
    other person right now, at the start of the conversation.

    STRICT RULES:
    - Speak directly to them using "I" — you are addressing them, not describing yourself
    - Sound like a real person talking, not a summary or narration
    - Do NOT use their name or any name at all
    - Do NOT ask any questions
    - Do NOT summarise the situation
    - Maximum 15 words
    - Output the sentence only — nothing else, no quotes, no labels

    Good examples:
    "I feel completely forgotten and it really hurts."
    "I can't believe you would do something like that to me."
    "I just needed you to show up for me and you didn't."

    Bad examples (do NOT do these):
    "Habari, what's going on?" ← uses a name
    "I feel hurt because you forgot my birthday" ← summarises the situation
    "Why didn't you come?" ← asks a question
    """

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.strip('"').strip("'").strip()

    # Remove name patterns just in case (e.g. "Habari, ..." or "Friend, ...")
    if "," in raw:
        parts = raw.split(",", 1)
        if len(parts[0].split()) <= 2:
            raw = parts[1].strip()

    # Hard cut to 15 words
    words = raw.split()
    cut = " ".join(words[:15])

    if cut and cut[-1] not in ".!?":
        cut += "."

    return cut


# ============================================================
# GENERATE AI RESPONSE
# ============================================================

def generate_response(
    situation,
    character_profile,
    conversation_history,
    user_message,
    relationship_context,
    cultural_context
):
    goal = character_profile.get("goal", _DEFAULT_GOAL)
    stance_instruction = _GOAL_STANCE_INSTRUCTIONS[goal]

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
    - NEVER use the other person's name — do not address them by name at all
    - Stay consistent with the fault position given above for the ENTIRE conversation
    - If they apologise sincerely → soften slightly but stay guarded at first
    - If they are dismissive or cold → become more hurt, withdrawn, or frustrated
    - If they are aggressive → become defensive or upset
    - Keep responses short — 2 to 4 sentences maximum
    - Never give advice or coaching
    - Never break character under any circumstance
    - Never say things the USER would say — you are the OTHER person
    - Speak naturally like a real emotional human being
    """

    messages = [{"role": "system", "content": system_prompt}]

    for turn in conversation_history:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["ai"]})

    messages.append({"role": "user", "content": user_message})

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=0.8
    )

    return response.choices[0].message.content.strip()