"""
Multi-round debate orchestrator.

Runs agents through a structured 3-round debate:
  Round 1 (opening)   — each agent answers independently
  Round 2 (rebuttal)  — each agent reads all Round 1 answers and responds
  Round 3 (closing)   — each agent reads the full debate history and gives final synthesis

Within each round, agents run concurrently. Between rounds, we wait for all
agents to finish before proceeding (agents need the previous round's outputs).
"""
from concurrent.futures import ThreadPoolExecutor
from agents import run_agent_debate_round

ROUND_SEQUENCE = ["opening", "rebuttal", "closing"]


def run_debate(
    chosen_agents: list,
    question: str,
    chunks,
    num_rounds: int = 3,
    max_workers: int = 4,
) -> dict:
    """
    Orchestrate a multi-round debate among the chosen agents.

    Returns:
    {
        "rounds": [
            {"round_name": "opening", "round_number": 1, "responses": [...]},
            {"round_name": "rebuttal", "round_number": 2, "responses": [...]},
            {"round_name": "closing", "round_number": 3, "responses": [...]},
        ],
        "final_answers": [  # the closing-round responses (or last available)
            {"agent": key, "label": str, "answer": str, ...}, ...
        ],
        "all_agent_results": [  # flat list of all responses across all rounds (for backward compat)
            ...
        ],
    }
    """
    num_rounds = max(1, min(num_rounds, len(ROUND_SEQUENCE)))
    rounds_to_run = ROUND_SEQUENCE[:num_rounds]

    completed_rounds = []
    all_responses = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for round_idx, round_type in enumerate(rounds_to_run):
            # Submit all agents for this round concurrently
            futures = []
            for agent_key in chosen_agents:
                future = executor.submit(
                    run_agent_debate_round,
                    agent_key,
                    question,
                    chunks,
                    round_type,
                    completed_rounds,  # pass all completed rounds as context
                )
                futures.append(future)

            # Wait for all agents in this round to finish
            round_responses = [f.result() for f in futures]

            round_data = {
                "round_name": round_type,
                "round_number": round_idx + 1,
                "responses": round_responses,
            }
            completed_rounds.append(round_data)
            all_responses.extend(round_responses)

    # The final answers are the last round's responses
    final_round = completed_rounds[-1]
    final_answers = [
        r for r in final_round["responses"] if r.get("answer")
    ]

    # If the final round produced no answers, fall back to the latest round that did
    if not final_answers:
        for prev_round in reversed(completed_rounds[:-1]):
            final_answers = [r for r in prev_round["responses"] if r.get("answer")]
            if final_answers:
                break

    return {
        "rounds": completed_rounds,
        "final_answers": final_answers,
        "all_agent_results": all_responses,
    }
