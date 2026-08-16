"""
The combiner agent. Takes all competing answers plus the verifier's scores
and produces one final, polished answer.

In debate mode, the combiner receives the full debate transcript and leans
on the closing arguments (Round 3) while incorporating the best insights
from the entire debate. The best communicator's points get extra weight.
"""
from gemini_client import generate, GeminiError
from agents import build_transcript_block


def combine_debate(question: str, debate_data: dict, judge_result: dict) -> str:
    """
    Synthesize a final answer from a full debate.
    debate_data: output from debate.run_debate()
    judge_result: output from verifier.judge_debate()
    """
    rounds = debate_data.get("rounds", [])
    final_answers = debate_data.get("final_answers", [])

    if not final_answers:
        return "No agent was able to produce an answer for this question."

    scores = judge_result.get("scores", {})
    winner = judge_result.get("winner")
    best_comm = judge_result.get("best_communicator")

    # Build closing argument blocks with scores
    closing_blocks = []
    for r in final_answers:
        s = scores.get(r["agent"], {})
        overall = s.get("overall", "n/a")
        engagement = s.get("engagement", "n/a")
        persuasiveness = s.get("persuasiveness", "n/a")
        tags = []
        if r["agent"] == winner:
            tags.append("WINNER")
        if r["agent"] == best_comm:
            tags.append("BEST COMMUNICATOR")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        closing_blocks.append(
            f'{r["label"]}{tag_str} [overall: {overall}/10, '
            f'engagement: {engagement}/10, persuasiveness: {persuasiveness}/10]:\n{r["answer"]}'
        )
    closing_block = "\n\n".join(closing_blocks)

    # Also include a summary of the debate flow
    debate_summary = build_transcript_block(rounds)

    prompt = (
        f"QUESTION: {question}\n\n"
        f"A structured debate took place between research agents over {len(rounds)} rounds.\n\n"
        f"FULL DEBATE TRANSCRIPT:\n{debate_summary}\n\n"
        f"CLOSING ARGUMENTS (with verifier scores):\n{closing_block}\n\n"
        f"The verifier chose '{winner}' as the overall winner and '{best_comm}' as the best "
        f"communicator.\n\n"
        "Write ONE final answer that synthesizes the best of the entire debate. Guidelines:\n"
        "1. Lean most heavily on the winner's and best communicator's closing arguments.\n"
        "2. Incorporate any genuinely useful points that emerged DURING the debate — "
        "especially insights that agents discovered through cross-examination.\n"
        "3. If agents identified contradictions or gaps in the evidence, note them.\n"
        "4. Do NOT mention the agents, the competition, scores, or the debate itself — "
        "just give the best possible standalone answer, well-organized and directly useful."
    )

    try:
        return generate(prompt, temperature=0.4)
    except GeminiError as e:
        # Fall back to the winning agent's closing argument
        winner_answer = next(
            (r["answer"] for r in final_answers if r["agent"] == winner),
            final_answers[0]["answer"] if final_answers else "No answer available.",
        )
        return f"(Combiner unavailable: {e})\n\n{winner_answer}"


def combine(question: str, agent_results: list, judge_result: dict) -> str:
    """
    Legacy single-round combiner (backward compatible).
    """
    competing = [r for r in agent_results if r["answer"]]
    if not competing:
        return "No agent was able to produce an answer for this question."

    scores = judge_result.get("scores", {})
    winner = judge_result.get("winner")

    blocks = []
    for r in competing:
        s = scores.get(r["agent"], {})
        overall = s.get("overall", "n/a")
        tag = " (WINNER)" if r["agent"] == winner else ""
        blocks.append(f'{r["label"]}{tag} [verifier score: {overall}/10]:\n{r["answer"]}')
    answers_block = "\n\n".join(blocks)

    prompt = (
        f"QUESTION: {question}\n\n"
        f"Below are answers from several competing research agents, each with a verifier score.\n\n"
        f"{answers_block}\n\n"
        "Write ONE final answer that synthesizes the best of all of them. Lean most heavily on "
        "the highest-scoring agent(s), but fold in any genuinely useful point (a caveat from "
        "the skeptic, a clarity improvement from the explainer, etc.) that the top agent missed. "
        "Do not mention the agents, the competition, or scores in your answer -- just give the "
        "best possible standalone answer to the question, well-organized and directly useful to "
        "the reader."
    )

    try:
        return generate(prompt, temperature=0.4)
    except GeminiError as e:
        # Fall back to the winning agent's raw answer if the combiner call fails.
        winner_answer = next((r["answer"] for r in competing if r["agent"] == winner), competing[0]["answer"])
        return f"(Combiner unavailable: {e})\n\n{winner_answer}"
