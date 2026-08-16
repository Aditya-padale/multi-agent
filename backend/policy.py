"""
The reinforcement-learning piece.

This is framed as a multi-armed bandit, which is the honest, standard way to
do "which of N agents should I trust more" learning -- each agent is an
"arm", the verifier's score is the reward signal, and the policy updates
which agents get picked (and how much their vote counts) as evidence
accumulates. It's simpler than a deep policy network, but it's real online
reinforcement learning with a persisted, evolving policy -- not a canned
leaderboard.

Mechanics:
- Each agent has Beta(alpha, beta) parameters -- a standard Bayesian bandit
  (Thompson Sampling). Winning a round pushes alpha up; losing pushes beta up,
  scaled by how well the agent scored (partial credit, not just win/lose).
- To pick which agents answer a given question, we sample from each agent's
  Beta distribution (Thompson Sampling) and take the top K -- this naturally
  balances exploration (untested agents still get picked sometimes) with
  exploitation (proven agents get picked more often as their alpha grows).
- State is persisted to disk as JSON so learning survives server restarts.
"""
import json
import os
import random
from config import POLICY_STATE_PATH

AGENT_KEYS = ["literalist", "analyst", "skeptic", "explainer"]


class BanditPolicy:
    def __init__(self, path: str = POLICY_STATE_PATH):
        self.path = path
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                for k in AGENT_KEYS:
                    data.setdefault(k, {"alpha": 1.0, "beta": 1.0, "rounds": 0, "wins": 0})
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {k: {"alpha": 1.0, "beta": 1.0, "rounds": 0, "wins": 0} for k in AGENT_KEYS}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

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
        self._save()

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

    def reset(self):
        self.state = {k: {"alpha": 1.0, "beta": 1.0, "rounds": 0, "wins": 0} for k in AGENT_KEYS}
        self._save()
