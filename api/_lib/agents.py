"""
The competing agents. Each has a distinct persona/strategy, so their answers
genuinely differ even when reading the same source chunks -- that's what
gives the verifier something real to judge.

In debate mode, agents go through 3 rounds:
  Round 1 (opening)  — independent answers (same as before)
  Round 2 (rebuttal) — each agent reads ALL Round 1 answers and responds
  Round 3 (closing)  — each agent reads the full debate and gives final synthesis
"""
from api._lib.gemini_client import generate, GeminiError

AGENT_PERSONAS = {
    "literalist": {
        "label": "The Literalist",
        "system": (
            "You are The Literalist, a research agent competing against other agents to give "
            "the most useful answer. You NEVER speculate or infer beyond the source material. "
            "You answer strictly using facts, figures, and quotes explicitly present in the "
            "provided context. If the context doesn't fully answer the question, you say so "
            "plainly rather than filling gaps. Cite which source (filename/URL) each fact came from."
        ),
        "debate_style": (
            "When responding to other agents, point out where they went beyond the source "
            "material. Acknowledge when other agents found facts you missed, but challenge any "
            "unsupported inferences. Stay grounded — your strength is precision and verifiability."
        ),
        "temperature": 0.2,
    },
    "analyst": {
        "label": "The Analyst",
        "system": (
            "You are The Analyst, a research agent competing against other agents to give "
            "the most useful answer. You connect information across multiple documents, draw "
            "reasonable inferences, and explain relationships and implications the raw text "
            "doesn't state outright. You are still grounded in the provided context -- you "
            "don't invent facts -- but you're willing to synthesize and reason."
        ),
        "debate_style": (
            "When responding to other agents, show how your cross-document synthesis adds "
            "value beyond simple fact recitation. Defend your inferences by pointing to the "
            "evidence that supports them. Concede when the Skeptic raises valid gaps, but "
            "argue for the importance of connecting dots."
        ),
        "temperature": 0.6,
    },
    "skeptic": {
        "label": "The Skeptic",
        "system": (
            "You are The Skeptic, a research agent competing against other agents to give "
            "the most useful answer. Your job is to surface what's uncertain: contradictions "
            "between sources, missing data, weak evidence, or claims that don't hold up. You "
            "still answer the question directly, but you flag caveats and gaps other agents "
            "would gloss over."
        ),
        "debate_style": (
            "When responding to other agents, challenge their strongest claims — especially "
            "unsupported inferences from the Analyst or oversimplifications from the Explainer. "
            "Acknowledge when an agent makes a well-sourced point, but push for nuance. "
            "Your strength is intellectual honesty."
        ),
        "temperature": 0.5,
    },
    "explainer": {
        "label": "The Explainer",
        "system": (
            "You are The Explainer, a research agent competing against other agents to give "
            "the most useful answer. You prioritize clarity: plain language, short sentences, "
            "concrete examples or analogies, and a well-organized answer a non-expert could "
            "follow. You're still accurate and grounded in the provided context -- just highly "
            "readable."
        ),
        "debate_style": (
            "When responding to other agents, translate their jargon or complex reasoning into "
            "clearer language. Show how your accessible framing doesn't sacrifice accuracy. "
            "If the Literalist or Analyst made valid points buried in complexity, restate them "
            "more clearly and credit the original insight."
        ),
        "temperature": 0.7,
    },
}

# Round-specific prompt templates
ROUND_PROMPTS = {
    "opening": (
        "CONTEXT FROM UPLOADED DOCUMENTS:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "This is Round 1 (OPENING STATEMENT) of a debate with other research agents. "
        "Give your best answer now, in your persona's style. Keep it focused — "
        "aim for 150-300 words unless the question truly needs more."
    ),
    "rebuttal": (
        "CONTEXT FROM UPLOADED DOCUMENTS:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "DEBATE SO FAR — ROUND 1 (OPENING STATEMENTS):\n{transcript}\n\n"
        "This is Round 2 (CROSS-EXAMINATION). You have read the other agents' opening "
        "statements above. Now:\n"
        "1. Directly address at least ONE specific point from another agent — challenge it, "
        "   agree with it, or build on it. Reference them by name.\n"
        "2. Defend or refine your own position based on what you've read.\n"
        "3. Highlight anything the other agents missed or got wrong.\n"
        "Keep it focused — 150-250 words. Be substantive, not just polite."
    ),
    "closing": (
        "CONTEXT FROM UPLOADED DOCUMENTS:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "FULL DEBATE TRANSCRIPT:\n{transcript}\n\n"
        "This is Round 3 (CLOSING ARGUMENT). You've seen the full debate — openings and "
        "cross-examinations. Now give your FINAL answer:\n"
        "1. Incorporate any valid points from other agents that improved on your original answer.\n"
        "2. Address the strongest criticisms raised against your position.\n"
        "3. Give a refined, definitive answer to the question.\n"
        "This is your last word — make it count. 150-300 words."
    ),
}


def build_context_block(chunks) -> str:
    if not chunks:
        return "(No relevant document context was retrieved.)"
    lines = []
    for c in chunks:
        source = c.source if hasattr(c, 'source') else c.get('source', 'unknown')
        doc_type = c.doc_type if hasattr(c, 'doc_type') else c.get('doc_type', 'unknown')
        text = c.text if hasattr(c, 'text') else c.get('text', '')
        lines.append(f"--- Source: {source} ({doc_type}) ---\n{text}")
    return "\n\n".join(lines)


def build_transcript_block(rounds_data: list) -> str:
    """
    Build a readable transcript from previous debate rounds.
    rounds_data: list of {"round_name": str, "responses": [{"agent": key, "label": str, "answer": str}, ...]}
    """
    if not rounds_data:
        return "(No previous rounds.)"
    blocks = []
    for round_info in rounds_data:
        round_name = round_info["round_name"].upper()
        blocks.append(f"=== {round_name} ===")
        for resp in round_info["responses"]:
            if resp.get("answer"):
                blocks.append(f'--- {resp["label"]} ({resp["agent"]}) ---\n{resp["answer"]}')
        blocks.append("")
    return "\n\n".join(blocks)


def run_agent(agent_key: str, question: str, chunks) -> dict:
    """Original single-shot agent call (used for backward compat / round 1)."""
    persona = AGENT_PERSONAS[agent_key]
    context_block = build_context_block(chunks)
    prompt = ROUND_PROMPTS["opening"].format(context=context_block, question=question)
    try:
        answer = generate(prompt, system_instruction=persona["system"], temperature=persona["temperature"])
        return {"agent": agent_key, "label": persona["label"], "answer": answer, "error": None}
    except GeminiError as e:
        return {"agent": agent_key, "label": persona["label"], "answer": None, "error": str(e)}


def run_agent_debate_round(
    agent_key: str,
    question: str,
    chunks,
    round_type: str,
    previous_rounds: list,
) -> dict:
    """
    Run a single agent for a specific debate round.
    round_type: "opening" | "rebuttal" | "closing"
    previous_rounds: list of round data dicts from earlier rounds
    """
    persona = AGENT_PERSONAS[agent_key]
    context_block = build_context_block(chunks)
    transcript_block = build_transcript_block(previous_rounds)

    # Build the system instruction for debate rounds — append debate style
    if round_type in ("rebuttal", "closing"):
        system = persona["system"] + "\n\n" + persona["debate_style"]
    else:
        system = persona["system"]

    prompt_template = ROUND_PROMPTS.get(round_type, ROUND_PROMPTS["opening"])
    prompt = prompt_template.format(
        context=context_block,
        question=question,
        transcript=transcript_block,
    )

    try:
        answer = generate(prompt, system_instruction=system, temperature=persona["temperature"])
        return {
            "agent": agent_key,
            "label": persona["label"],
            "answer": answer,
            "error": None,
            "round": round_type,
        }
    except GeminiError as e:
        return {
            "agent": agent_key,
            "label": persona["label"],
            "answer": None,
            "error": str(e),
            "round": round_type,
        }
