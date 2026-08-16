"""
POST /api/ask — Run the full debate pipeline.
Vercel serverless function.

Receives chunks + policy state from the client, runs the debate, judge,
combiner pipeline, and returns results + updated policy state.

This is the heaviest endpoint — it makes multiple Gemini API calls.
maxDuration is set to 60s (Hobby) — increase if on Vercel Pro.
"""
from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            question = body.get("question", "").strip()
            if not question:
                self._error(400, "Question is empty.")
                return

            chunks_data = body.get("chunks", [])
            if not chunks_data:
                self._error(400, "No documents uploaded yet.")
                return

            num_agents = body.get("num_agents", 3)
            debate_rounds = body.get("debate_rounds", 3)
            policy_state = body.get("policy_state", None)

            # Import here to reduce cold-start time for other endpoints
            from api._lib.ingestion import DocStore
            from api._lib.agents import AGENT_PERSONAS
            from api._lib.verifier import judge, judge_debate, normalized_rewards
            from api._lib.combiner import combine, combine_debate
            from api._lib.policy import BanditPolicy, AGENT_KEYS
            from api._lib.debate import run_debate

            # Reconstruct state from client data
            store = DocStore.from_chunks_list(chunks_data)
            policy = BanditPolicy(state=policy_state)

            k = max(3, min(num_agents, len(AGENT_KEYS)))  # min 3 for debates
            chosen_agents = policy.select_agents(k=k)
            chunks = store.retrieve(question, top_k=8)
            num_rounds = max(1, min(debate_rounds, 3))

            # Run the multi-round debate
            debate_data = run_debate(
                chosen_agents=chosen_agents,
                question=question,
                chunks=chunks,
                num_rounds=num_rounds,
            )

            # Judge the debate
            if num_rounds > 1:
                judge_result = judge_debate(question, chunks, debate_data)
            else:
                # Single round — use legacy judge for backward compat
                agent_results = debate_data["final_answers"]
                judge_result = judge(question, chunks, agent_results)
                # Add best_communicator for consistency
                if "best_communicator" not in judge_result:
                    judge_result["best_communicator"] = judge_result.get("winner")

            rewards = normalized_rewards(judge_result["scores"]) if judge_result["scores"] else {}

            # Combine the final answer
            if num_rounds > 1:
                final_answer = combine_debate(question, debate_data, judge_result)
            else:
                final_answer = combine(question, debate_data["final_answers"], judge_result)

            if rewards and judge_result.get("winner"):
                policy.update(rewards, judge_result["winner"])

            record = {
                "question": question,
                "participants": chosen_agents,
                "debate_rounds": num_rounds,
                "debate": debate_data["rounds"],       # full debate transcript
                "agent_results": debate_data["final_answers"],  # closing arguments (backward compat)
                "judge": judge_result,
                "rewards": rewards,
                "final_answer": final_answer,
                "leaderboard_after": policy.leaderboard(),
                "policy_state": policy.to_dict(),  # updated state for client to persist
            }

            self._json_response(200, record)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error(500, f"Debate pipeline error: {e}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, detail):
        self._json_response(status, {"detail": detail})
