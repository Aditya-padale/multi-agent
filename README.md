# ARENA — Multi-Document Research Agent

Upload PDFs, CSVs, Excel files, or webpages. Ask a question. Four differently-styled
agents read the same evidence and independently answer. A verifier agent judges every
answer on a rubric and picks a winner. A combiner agent synthesizes the final response.
A persistent reinforcement-learning policy tracks which agents to trust more over time.

## ⚠️ Before you do anything else

You pasted a live Gemini API key in our chat. **Treat it as compromised** — go to
[Google AI Studio](https://aistudio.google.com/apikey), delete/regenerate that key, and
use the new one only via environment variables (never paste keys into a chat again).

## Architecture

```
backend/
  main.py         FastAPI app — upload, ask, leaderboard endpoints
  ingestion.py     PDF/CSV/XLSX/webpage → text chunks, TF-IDF retrieval
  agents.py        4 competing agent personas (literalist/analyst/skeptic/explainer)
  verifier.py      Judge agent — scores each answer, picks a winner
  combiner.py      Synthesizes the final answer from all competing answers
  policy.py        Persistent multi-armed bandit (Thompson Sampling) — the RL piece
  gemini_client.py Thin REST wrapper around Gemini's generateContent
  config.py        Reads GEMINI_API_KEY / GEMINI_MODEL from environment
frontend/
  index.html / style.css / app.js   Plain HTML/CSS/JS UI, no build step needed
```

### How a question flows through the system

1. **Retrieval** — your question is matched against uploaded document chunks via
   TF-IDF + cosine similarity (fast, no embedding API needed).
2. **Selection (RL)** — the bandit policy uses Thompson Sampling to pick which
   agents compete this round, weighted by their learned trust score. Untested
   agents still get picked sometimes (exploration); proven agents get picked more
   as evidence accumulates (exploitation).
3. **Competition** — the chosen agents each independently answer using the same
   retrieved context, in their own style:
   - **The Literalist** — sticks strictly to quoted facts, flags what's not covered
   - **The Analyst** — connects facts across documents, draws inferences
   - **The Skeptic** — surfaces contradictions, gaps, and weak evidence
   - **The Explainer** — prioritizes plain-language clarity
4. **Verification** — a judge agent scores every answer on grounding, completeness,
   and clarity (0–10 each), and picks a winner.
5. **Reward + learning** — each agent's normalized score becomes its reward. The
   bandit's Beta(alpha, beta) parameters update accordingly and are saved to
   `backend/policy_state.json`, so the policy keeps learning across restarts.
6. **Combination** — a combiner agent writes one final answer, leaning on the
   highest-scoring agents but folding in anything genuinely useful the others caught.

### Why a bandit instead of a neural policy

A multi-armed bandit (Thompson Sampling over Beta distributions) is the standard,
honest way to do "learn which of N arms to trust" with a scalar reward signal like a
verifier's score. It's real online reinforcement learning — not a fixed leaderboard —
and it persists and keeps improving, without needing a training pipeline, GPU, or
labeled dataset a policy-gradient network would require. You can swap in something
heavier later (e.g. a contextual bandit that also looks at question type) using the
same `policy.py` interface.

## Setup

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp ../.env.example ../.env
# edit .env and paste your NEW (rotated) Gemini key
export $(grep -v '^#' ../.env | xargs)   # or use python-dotenv / your shell's env loading

uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the FastAPI app serves the frontend directly, no
separate dev server needed.

## Notes

- This is a single-session MVP: one shared document store and one shared policy file.
  For multi-user deployment, key `store` and `history` in `main.py` by a session ID
  from the frontend (e.g. a cookie), and give each session its own bandit state.
- Retrieval is lexical (TF-IDF), not semantic. For better recall on paraphrased
  questions, swap `ingestion.py`'s retrieval for real embeddings (Gemini has an
  embedding endpoint) — the `DocStore` interface won't need to change.
- I built and smoke-tested every endpoint (upload, ask, leaderboard, reset) in this
  sandbox, including confirming the app degrades gracefully without an API key. I
  could not make a live Gemini call end-to-end here because this sandbox's network
  is restricted to package registries (pypi/npm/github) — `generativelanguage.googleapis.com`
  isn't reachable from it. Test the live agent calls on your machine once your key is set.
# multi-agent
