"""
The reinforcement-learning piece — stateless version for serverless.

This is framed as a multi-armed bandit, which is the honest, standard way to
do "which of N agents should I trust more" learning -- each agent is an
"arm", the verifier's score is the reward signal, and the policy updates
which agents get picked (and how much their vote counts) as evidence
accumulates.

Adapted for serverless: no filesystem persistence. State is sent by the client
(stored in localStorage) and returned after updates.
"""
import random

AGENT_KEYS = ["literalist", "analyst", "skeptic", "explainer"]

DEFAULT_STATE = {k: {"alpha": 1.0, "beta": 1.0, "rounds": 0, "wins": 0} for k in AGENT_KEYS}


class BanditPolicy:
    def __init__(self, state: dict | None = None):
        """Initialize from client-provided state dict, or use defaults."""
        if state:
            self.state = dict(state)
            # Ensure all agent keys exist
            for k in AGENT_KEYS:
                self.state.setdefault(k, {"alpha": 1.0, "beta": 1.0, "rounds": 0, "wins": 0})
        else:
            self.state = {k: {"alpha": 1.0, "beta": 1.0, "rounds": 0, "wins": 0} for k in AGENT_KEYS}

    def trust_score(self, agent_key: str) -> float:
        """Mean of the agent's Beta distribution -- its current 'trust' in [0,1]."""
        s = self.state[agent_key]
        return s["alpha"] / (s["alpha"] + s["beta"])

    def select_agents(self, k: int = 3) -> list:
        """Thompson Sampling: sample a value per agent, return the top-k agent keys."""
        samples = []
        for key in AGENT_KEYS:
            s = self.state[key]
            sample = random.betavariate(s["alpha"], s["beta"])
            samples.append((sample, key))
        samples.sort(reverse=True)
        chosen = [key for _, key in samples[:k]]
        return chosen

    def update(self, scores: dict, winner: str):
        """
        scores: {agent_key: normalized_score_0_to_1} for every agent that competed this round
        winner: agent_key of the verifier's chosen winner
        """
        for key, score in scores.items():
            s = self.state[key]
            s["rounds"] += 1
            if key == winner:
                s["wins"] += 1
            # Reward-weighted Beta update: strong scores push alpha up more,
            # weak scores push beta up more. This gives partial credit instead
            # of a harsh binary win/lose signal.
            s["alpha"] += score
            s["beta"] += (1 - score)

    def leaderboard(self) -> list:
        rows = []
        for key in AGENT_KEYS:
            s = self.state[key]
            rows.append({
                "agent": key,
                "trust_score": round(self.trust_score(key), 4),
                "rounds": s["rounds"],
                "wins": s["wins"],
                "win_rate": round(s["wins"] / s["rounds"], 3) if s["rounds"] else None,
            })
        rows.sort(key=lambda r: r["trust_score"], reverse=True)
        return rows

    def to_dict(self) -> dict:
        """Serialize state for client storage."""
        return dict(self.state)

    def reset(self):
        self.state = {k: {"alpha": 1.0, "beta": 1.0, "rounds": 0, "wins": 0} for k in AGENT_KEYS}
