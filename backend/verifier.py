"""
The verifier/judge agent. Scores every competing agent's answer against the
retrieved context on a fixed rubric, picks a winner, and returns normalized
scores that feed the RL policy's reward signal.

Enhanced for debate mode: now judges agents on 5 criteria including how well
they engaged with other agents' arguments (engagement) and how persuasively
they defended/evolved their position across rounds (persuasiveness).
"""
import json
import re
import sys
from gemini_client import generate, GeminiError
from agents import build_context_block, build_transcript_block

JUDGE_MODEL = "gemini-2.5-flash-lite"

RUBRIC_DEBATE = (
    "Score each agent 0-10 on these FIVE criteria based on their FULL performance "
    "across all debate rounds, then compute an overall score:\n"
    "1. GROUNDING (0-10): Is the answer actually supported by the provided context, with no "
    "invented facts?\n"
    "2. COMPLETENESS (0-10): Does it fully address the question, including relevant nuance?\n"
    "3. CLARITY (0-10): Is it well-organized and easy to understand?\n"
    "4. ENGAGEMENT (0-10): How well did the agent engage with OTHER agents' arguments? "
    "Did they directly address counter-points, acknowledge valid criticisms, and build on "
    "others' insights? An agent that ignored the debate scores low here.\n"
    "5. PERSUASIVENESS (0-10): How convincingly did the agent defend and evolve their position "
    "across rounds? Did their final answer improve from incorporating debate feedback? "
    "An agent whose closing argument is stronger than their opening scores high.\n"
    "\nOVERALL should be a weighted judgment: engagement and persuasiveness matter MORE than "
    "in a normal assessment because this is a debate. The best communicator should win.\n"
)

RUBRIC_SIMPLE = (
    "Score each answer 0-10 on these three criteria, then compute an overall score:\n"
    "1. GROUNDING (0-10): Is the answer actually supported by the provided context, with no "
    "invented facts?\n"
    "2. COMPLETENESS (0-10): Does it fully address the question, including relevant nuance?\n"
    "3. CLARITY (0-10): Is it well-organized and easy to understand?\n"
)


def judge_debate(question: str, chunks, debate_data: dict) -> dict:
    """
    Judge a full multi-round debate.

    debate_data: output from debate.run_debate() — contains "rounds" and "final_answers"
    Returns: {
        "scores": {agent_key: {"grounding":.., "completeness":.., "clarity":..,
                                "engagement":.., "persuasiveness":.., "overall":..}},
        "winner": agent_key,
        "best_communicator": agent_key,  # highest engagement + persuasiveness
        "rationale": str,
        "raw_ok": bool
    }
    """
    context_block = build_context_block(chunks)
    rounds = debate_data.get("rounds", [])
    final_answers = debate_data.get("final_answers", [])

    if not final_answers:
        return {
            "scores": {}, "winner": None, "best_communicator": None,
            "rationale": "No agent produced an answer.", "raw_ok": False,
        }

    # Build the full debate transcript for the judge
    transcript_block = build_transcript_block(rounds)

    # Build a list of competing agents from the final answers
    agent_keys = [r["agent"] for r in final_answers]

    prompt = (
        f"You are an impartial verifier judging a DEBATE between research agents.\n"
        f"The agents competed in {len(rounds)} round(s). You must evaluate their FULL "
        f"performance across all rounds — not just their final answers.\n\n"
        f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"FULL DEBATE TRANSCRIPT:\n{transcript_block}\n\n"
        f"{RUBRIC_DEBATE}\n"
        "IMPORTANT RULES:\n"
        "- Do NOT give tied overall scores. Each agent must have a unique overall score.\n"
        "- The 'winner' is the agent with the highest overall score.\n"
        "- The 'best_communicator' is the agent with the highest combined engagement + persuasiveness.\n"
        "  If that differs from the winner, note it.\n\n"
        "Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:\n"
        "{\n"
        '  "scores": {\n'
        '    "<agent_key>": {"grounding": <0-10>, "completeness": <0-10>, "clarity": <0-10>, '
        '"engagement": <0-10>, "persuasiveness": <0-10>, "overall": <0-10>},\n'
        "    ...\n"
        "  },\n"
        '  "winner": "<agent_key with highest overall>",\n'
        '  "best_communicator": "<agent_key with highest engagement + persuasiveness>",\n'
        '  "rationale": "<3-4 sentences: why the winner won, and who was the best communicator and why>"\n'
        "}\n"
        f"Use the exact agent_key strings: {', '.join(agent_keys)}."
    )

    raw = None
    try:
        raw = generate(prompt, temperature=0.1, model=JUDGE_MODEL)
        parsed = _parse_judge_response(raw)
        result = _validate_debate_result(parsed, agent_keys)
        return result
    except (GeminiError, json.JSONDecodeError, ValueError, KeyError) as e:
        if raw:
            try:
                parsed = _aggressive_json_repair(raw)
                result = _validate_debate_result(parsed, agent_keys)
                print(f"[verifier] Repaired judge JSON on second pass.", file=sys.stderr)
                return result
            except Exception as repair_error:
                print(f"[verifier] Repair failed: {repair_error}", file=sys.stderr)

        # Fallback: if judging fails, degrade gracefully with differentiated scores
        fallback_scores = {}
        for i, r in enumerate(final_answers):
            # Give slightly different fallback scores so there's always a clear winner
            base = 5 + (len(final_answers) - i) * 0.1
            fallback_scores[r["agent"]] = {
                "grounding": round(base, 1), "completeness": round(base, 1),
                "clarity": round(base, 1), "engagement": round(base, 1),
                "persuasiveness": round(base, 1), "overall": round(base, 1),
            }
        print(f"[verifier] Falling back to default scores. Initial error: {e}", file=sys.stderr)
        return {
            "scores": fallback_scores,
            "winner": final_answers[0]["agent"],
            "best_communicator": final_answers[0]["agent"],
            "rationale": "Verifier temporarily unavailable. Defaulted to first responding agent.",
            "raw_ok": False,
        }


def judge(question: str, chunks, agent_results: list) -> dict:
    """
    Legacy single-round judge (backward compatible).
    agent_results: list of {"agent": key, "label": str, "answer": str|None, "error": str|None}
    Returns: {
        "scores": {agent_key: {"grounding":.., "completeness":.., "clarity":.., "overall":..}},
        "winner": agent_key,
        "rationale": str,
        "raw_ok": bool
    }
    """
    context_block = build_context_block(chunks)
    competing = [r for r in agent_results if r["answer"]]

    if not competing:
        return {"scores": {}, "winner": None, "rationale": "No agent produced an answer.", "raw_ok": False}

    answers_block = "\n\n".join(
        f'AGENT "{r["agent"]}" ({r["label"]}):\n{r["answer"]}' for r in competing
    )

    prompt = (
        f"You are an impartial verifier judging a competition between research agents.\n\n"
        f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"CANDIDATE ANSWERS:\n{answers_block}\n\n"
        f"{RUBRIC_SIMPLE}\n"
        "IMPORTANT: Do NOT give tied overall scores. Each agent must have a unique overall score.\n\n"
        "Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:\n"
        "{\n"
        '  "scores": {\n'
        '    "<agent_key>": {"grounding": <0-10>, "completeness": <0-10>, "clarity": <0-10>, "overall": <0-10>},\n'
        "    ...\n"
        "  },\n"
        '  "winner": "<agent_key with highest overall>",\n'
        '  "rationale": "<2-3 sentences on why the winner won>"\n'
        "}\n"
        "Use the exact agent_key strings shown above (e.g. literalist, analyst, skeptic, explainer)."
    )

    raw = None
    try:
        raw = generate(prompt, temperature=0.1, model=JUDGE_MODEL)
        parsed = _parse_judge_response(raw)
        scores = parsed.get("scores", {})
        winner = parsed.get("winner")
        rationale = parsed.get("rationale", "")
        if not scores or winner not in scores:
            raise ValueError("Judge response missing scores or valid winner")
        return {"scores": scores, "winner": winner, "rationale": rationale, "raw_ok": True}
    except (GeminiError, json.JSONDecodeError, ValueError, KeyError) as e:
        # Try repair: extract the best JSON-like block and clean it up.
        if raw:
            try:
                parsed = _aggressive_json_repair(raw)
                scores = parsed.get("scores", {})
                winner = parsed.get("winner")
                rationale = parsed.get("rationale", "")
                if scores and winner in scores:
                    print(f"[verifier] Repaired judge JSON on second pass.", file=sys.stderr)
                    return {"scores": scores, "winner": winner, "rationale": rationale, "raw_ok": True}
            except Exception as repair_error:
                print(f"[verifier] Repair failed: {repair_error}", file=sys.stderr)

        # Fallback: if judging fails, don't crash the whole request -- degrade gracefully.
        fallback_scores = {}
        for i, r in enumerate(competing):
            base = 5 + (len(competing) - i) * 0.1
            fallback_scores[r["agent"]] = {
                "grounding": round(base, 1), "completeness": round(base, 1),
                "clarity": round(base, 1), "overall": round(base, 1),
            }
        print(f"[verifier] Falling back to default scores. Initial error: {e}", file=sys.stderr)
        return {
            "scores": fallback_scores,
            "winner": competing[0]["agent"],
            "rationale": "Verifier temporarily unavailable. Defaulted to first responding agent.",
            "raw_ok": False,
        }


def normalized_rewards(scores: dict) -> dict:
    """Convert the judge's 0-10 'overall' scores into 0-1 rewards for the bandit policy."""
    return {k: max(0.0, min(1.0, v.get("overall", 5) / 10.0)) for k, v in scores.items()}


def _validate_debate_result(parsed: dict, agent_keys: list) -> dict:
    """Validate and normalize a parsed judge response for debate mode."""
    scores = parsed.get("scores", {})
    winner = parsed.get("winner")
    best_communicator = parsed.get("best_communicator")
    rationale = parsed.get("rationale", "")

    if not scores or winner not in scores:
        raise ValueError("Judge response missing scores or valid winner")

    # Ensure all competing agents have scores
    for key in agent_keys:
        if key not in scores:
            scores[key] = {
                "grounding": 5, "completeness": 5, "clarity": 5,
                "engagement": 5, "persuasiveness": 5, "overall": 5,
            }

    # Compute best_communicator if missing or invalid
    if not best_communicator or best_communicator not in scores:
        comm_scores = {
            k: v.get("engagement", 0) + v.get("persuasiveness", 0)
            for k, v in scores.items()
        }
        best_communicator = max(comm_scores, key=comm_scores.get)

    # Break ties in overall scores — ensure unique ordering
    overall_scores = [(k, v.get("overall", 0)) for k, v in scores.items()]
    overall_scores.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    for i, (key, score) in enumerate(overall_scores):
        while score in seen:
            score = max(0, score - 0.1)
            scores[key]["overall"] = round(score, 1)
        seen.add(score)

    return {
        "scores": scores,
        "winner": winner,
        "best_communicator": best_communicator,
        "rationale": rationale,
        "raw_ok": True,
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    return text


def _extract_json_object(text: str) -> str:
    """Extract the largest valid JSON object from text, handling nested braces."""
    text = _strip_fences(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("Unmatched braces in response")
    return text[start : end + 1]


def _parse_judge_response(text: str) -> dict:
    """Parse judge response, trying multiple repair strategies."""
    candidate = _extract_json_object(text)

    # Try raw parse first.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Repair strategy 1: Remove trailing commas before closing braces/brackets.
    try:
        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair strategy 2: Fix unquoted keys (common LLM mistake).
    try:
        repaired = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', candidate)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair strategy 3: Fix single quotes to double quotes.
    try:
        repaired = candidate.replace("'", '"')
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair strategy 4: Strip control characters.
    try:
        repaired = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', candidate)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # If all else fails, raise the original error.
    raise json.JSONDecodeError("All repair strategies exhausted", candidate, 0)


def _aggressive_json_repair(text: str) -> dict:
    """Aggressively repair JSON: try multiple extraction + parsing strategies."""
    # Strategy 1: Normal extraction and parse.
    try:
        return _parse_judge_response(text)
    except Exception:
        pass

    # Strategy 2: Look for JSON-like patterns and try to construct a valid object.
    try:
        scores_match = re.search(r'"scores"\s*:\s*(\{.+?\})\s*\}', text, re.DOTALL)
        winner_match = re.search(r'"winner"\s*:\s*"([^"]+)"', text)
        best_comm_match = re.search(r'"best_communicator"\s*:\s*"([^"]+)"', text)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)

        if scores_match and winner_match:
            scores_text = scores_match.group(1) + "}"
            obj = {
                "scores": {},
                "winner": winner_match.group(1),
                "best_communicator": best_comm_match.group(1) if best_comm_match else winner_match.group(1),
                "rationale": rationale_match.group(1) if rationale_match else "",
            }

            # Parse individual score entries.
            for match in re.finditer(r'"([^"]+)"\s*:\s*\{([^}]+)\}', scores_text):
                agent_key = match.group(1)
                scores_block = match.group(2)
                entry = {}
                for score_match in re.finditer(r'"(\w+)"\s*:\s*(\d+(?:\.\d+)?)', scores_block):
                    entry[score_match.group(1)] = float(score_match.group(2))
                if entry:
                    obj["scores"][agent_key] = entry

            if obj["scores"]:
                return obj
    except Exception:
        pass

    # Strategy 3: Try to extract scores line-by-line with regex
    try:
        obj = {"scores": {}, "winner": "", "rationale": ""}
        # Find agent scores blocks
        agent_pattern = r'"(\w+)"\s*:\s*\{[^}]*"overall"\s*:\s*(\d+(?:\.\d+)?)[^}]*\}'
        for match in re.finditer(agent_pattern, text):
            agent_key = match.group(1)
            overall = float(match.group(2))
            # Try to get all individual scores
            block = match.group(0)
            entry = {}
            for score_match in re.finditer(r'"(\w+)"\s*:\s*(\d+(?:\.\d+)?)', block):
                sname = score_match.group(1)
                if sname != agent_key:
                    entry[sname] = float(score_match.group(2))
            if entry:
                obj["scores"][agent_key] = entry

        winner_match = re.search(r'"winner"\s*:\s*"([^"]+)"', text)
        if winner_match:
            obj["winner"] = winner_match.group(1)

        best_comm_match = re.search(r'"best_communicator"\s*:\s*"([^"]+)"', text)
        if best_comm_match:
            obj["best_communicator"] = best_comm_match.group(1)

        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
        if rationale_match:
            obj["rationale"] = rationale_match.group(1)

        if obj["scores"] and obj["winner"]:
            return obj
    except Exception:
        pass

    raise ValueError("Could not extract or repair JSON from judge response")
