# conversation_engine.py
# Uses Groq (Llama 3.3 70B) to simulate the other person
# in a realistic, emotionally human conversation
# ============================================================

from groq import Groq

# ============================================================
# GROQ CLIENT
# ============================================================

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_client = Groq(api_key=GROQ_API_KEY)
_MODEL  = "llama-3.3-70b-versatile"


# ============================================================
# BUILD CHARACTER PROFILE
#
# Reads the situation the user described and creates a
# character for the AI to play throughout the conversation.
# ============================================================

def build_character_profile(situation, relationship_context, cultural_context):
    """
    Uses Groq to read the situation and generate a character profile.

    Input:
        situation            (str) — what the user described
        relationship_context (str) — elder / peer / subordinate / stranger
        cultural_context     (str) — african / western / asian

    Output:
        dict with:
            character_description (str) — who this person is
            emotional_state       (str) — how they feel right now
            opening_message       (str) — their first message to the user
            scenario_requires_apology (bool) — does this need an apology
    """

    prompt = f"""
    You are helping simulate a realistic emotional conversation for an EI training system.

    The user described this situation:
    "{situation}"

    Relationship: {relationship_context}
    Cultural context: {cultural_context}

    Your job is to create a character profile for the OTHER PERSON in this situation.
    Respond ONLY in this exact JSON format with no extra text:

    {{
        "character_description": "brief description of who this person is",
        "emotional_state": "how they are feeling right now in one phrase",
        "opening_message": "opening_message": "",
        "scenario_requires_apology": true or false
    }}

    RULES FOR opening_message:
    - Maximum 2 sentences. If you write 3, you have failed the instruction.
    - No questions whatsoever — not even rhetorical ones.
    - No accusations like "you don't care" or "you never think of me".
    - No dialogue — the character is not yet addressing the user directly.
    - Pure internal emotion only: sadness, hurt, disappointment, confusion.
    """

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    import json
    raw = response.choices[0].message.content.strip()

    # Remove markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    profile = json.loads(raw)
    profile["opening_message"] = generate_opening_message(situation, profile)
    return profile


def generate_opening_message(situation, character_profile):
    """
    Generates a short opening emotional statement.
    Hard-capped at 15 words via post-processing.
    """

    prompt = f"""
    Situation: {situation}
    Emotion: {character_profile['emotional_state']}

    Write one sentence starting with "I" describing how this person feels right now.
    Sound like a real person talking, not a book or poem.
    Maximum 15 words. No questions. Output the sentence only. Nothing else.

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

    YOUR RELATIONSHIP TO THE PERSON YOU ARE TALKING TO:
    You are the {relationship_context} in a {cultural_context} context.
    The person messaging you is the one who caused this situation.
    You are responding TO them — not speaking AS them.

    STRICT RULES:
    - You are ALWAYS the other person — never the user
    - You speak from YOUR feelings and YOUR perspective only
    - You react emotionally to what they say to you
    - If they apologise sincerely → soften slightly but stay hurt
    - If they are dismissive or cold → become more hurt or withdrawn
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


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Step 1 — Build character profile
    profile = build_character_profile(
        situation="My friend is upset because I forgot their birthday",
        relationship_context="peer",
        cultural_context="african"
    )

    print("CHARACTER PROFILE:")
    for k, v in profile.items():
        print(f"  {k}: {v}")

    # Step 2 — Simulate a conversation
    history = []

    user_msg_1 = "Look I have been really busy lately okay, it is not a big deal."

    ai_response_1 = generate_response(
        situation="My friend is upset because I forgot their birthday",
        character_profile=profile,
        conversation_history=history,
        user_message=user_msg_1,
        relationship_context="peer",
        cultural_context="african"
    )

    print(f"\nUser : {user_msg_1}")
    print(f"AI   : {ai_response_1}")

    history.append({"user": user_msg_1, "ai": ai_response_1})

    user_msg_2 = "I am sorry, I should have remembered. That was wrong of me."

    ai_response_2 = generate_response(
        situation="My friend is upset because I forgot their birthday",
        character_profile=profile,
        conversation_history=history,
        user_message=user_msg_2,
        relationship_context="peer",
        cultural_context="african"
    )

    print(f"\nUser : {user_msg_2}")
    print(f"AI   : {ai_response_2}")