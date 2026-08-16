<div align="center">

# ◆ ARENA

### Multi-Agent Research Debate System

[![Powered by Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-E7B93C?style=for-the-badge)](LICENSE)

**Upload documents. Ask questions. Watch AI agents debate to find the best answer.**

Four AI agents with distinct reasoning styles compete in structured, multi-round debates — judged by an impartial verifier — with a reinforcement-learning policy that learns which agents to trust over time.

[Live Demo](#deployment) · [How It Works](#how-it-works) · [Quick Start](#quick-start) · [API Reference](#api-reference)

</div>

---

## ✨ Features

- **🔬 Multi-Agent Debate** — Four distinct AI personas (Literalist, Analyst, Skeptic, Explainer) argue over your documents in structured 3-round debates
- **⚖️ Impartial Verification** — A judge agent scores every answer on grounding, completeness, clarity, engagement, and persuasiveness
- **🧠 Reinforcement Learning** — A Thompson Sampling bandit policy learns which agents perform best and prioritizes them over time
- **📄 Multi-Format Ingestion** — Upload PDFs, CSVs, Excel files, or paste webpage URLs
- **🔍 TF-IDF Retrieval** — Fast lexical search over document chunks with no external embedding API required
- **⚡ Serverless Architecture** — Deployed on Vercel with stateless Python functions and client-side state management
- **🎯 Zero Build Step Frontend** — Plain HTML/CSS/JS with a dark, modern interface — no framework, no bundler

---

## 🏗️ How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Client)                      │
│  ┌─────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │ Chunks  │  │ Sources  │  │ Policy (localStorage)  │  │
│  │ (memory)│  │ (memory) │  │ (persists sessions)    │  │
│  └────┬────┘  └────┬─────┘  └───────────┬────────────┘  │
│       └────────────┼────────────────────┘                │
│                    │                                     │
└────────────────────┼─────────────────────────────────────┘
                     │ HTTPS
┌────────────────────┼─────────────────────────────────────┐
│              Vercel Edge Network                         │
│  ┌─────────────────┼───────────────────────────────────┐ │
│  │    Serverless Python Functions                      │ │
│  │                 │                                   │ │
│  │  /api/upload ──→ Parse file → Return chunks         │ │
│  │  /api/upload_url → Fetch URL → Return chunks        │ │
│  │  /api/ask ─────→ Debate → Judge → Combine           │ │
│  │                    │                                │ │
│  │        ┌───────────┼───────────┐                    │ │
│  │        ▼           ▼           ▼                    │ │
│  │   ┌────────┐ ┌──────────┐ ┌──────────┐             │ │
│  │   │ Agents │ │ Verifier │ │ Combiner │             │ │
│  │   │ (×4)   │ │ (Judge)  │ │          │             │ │
│  │   └───┬────┘ └────┬─────┘ └────┬─────┘             │ │
│  │       └───────────┼────────────┘                    │ │
│  │                   ▼                                 │ │
│  │          Google Gemini API                          │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Question Pipeline

Every question flows through six stages:

| Stage | What Happens |
|-------|-------------|
| **1. Retrieval** | Question is matched against uploaded document chunks via TF-IDF + cosine similarity |
| **2. Selection (RL)** | Thompson Sampling picks which agents compete, balancing exploration vs. exploitation |
| **3. Debate** | Agents argue in up to 3 rounds: Opening → Cross-Examination → Closing Arguments |
| **4. Verification** | Judge agent scores each agent on 5 criteria (0–10) and picks a winner |
| **5. Learning** | Normalized scores feed the bandit policy — agent trust evolves with every question |
| **6. Synthesis** | Combiner agent writes one final answer, weighted by the judge's scores |

### The Four Agents

| Agent | Strategy | Temperature |
|-------|----------|:-----------:|
| **🔍 The Literalist** | Strict facts only — quotes, figures, citations. Flags gaps instead of guessing. | 0.2 |
| **🔗 The Analyst** | Cross-document synthesis — connects dots, draws inferences, explains implications. | 0.6 |
| **⚠️ The Skeptic** | Adversarial review — surfaces contradictions, weak evidence, missing data. | 0.5 |
| **💡 The Explainer** | Clarity-first — plain language, analogies, structured for non-experts. | 0.7 |

### Multi-Round Debate Format

```
Round 1 — OPENING STATEMENTS
  Each agent answers independently in their own style.

Round 2 — CROSS-EXAMINATION  
  Agents read all Round 1 answers and respond:
  challenge, agree, or build on other agents' points.

Round 3 — CLOSING ARGUMENTS
  Agents read the full debate and give their refined final answer,
  incorporating valid criticisms and insights from the debate.
```

### Why Thompson Sampling?

The RL policy uses a **multi-armed bandit with Beta distributions** (Thompson Sampling) — the standard approach for "learn which of N options to trust" with scalar rewards. Each agent maintains `Beta(α, β)` parameters that update with every debate:

- **Strong scores** → α increases → agent gets selected more often
- **Weak scores** → β increases → agent gets selected less often
- **Untested agents** → high variance → still get explored

This is real online reinforcement learning that improves with every question — not a static leaderboard.

---

## 📁 Project Structure

```
research-agent/
├── api/                           # Vercel serverless Python functions
│   ├── health.py                  # GET  /api/health
│   ├── upload.py                  # POST /api/upload
│   ├── upload_url.py              # POST /api/upload-url
│   ├── ask.py                     # POST /api/ask
│   └── _lib/                      # Shared modules (not routed by Vercel)
│       ├── config.py              #   Environment-based configuration
│       ├── gemini_client.py       #   Gemini REST API wrapper with fallbacks
│       ├── agents.py              #   Agent personas, prompts, debate logic
│       ├── ingestion.py           #   PDF/CSV/XLSX/webpage → text chunks
│       ├── verifier.py            #   Judge agent with JSON repair strategies
│       ├── combiner.py            #   Final answer synthesis
│       ├── policy.py              #   Stateless Thompson Sampling bandit
│       └── debate.py              #   Multi-round debate orchestrator
│
├── public/                        # Static frontend (served by Vercel CDN)
│   ├── index.html                 #   App shell
│   ├── style.css                  #   Dark theme design system
│   └── app.js                     #   Client-side state & rendering
│
├── backend/                       # Legacy local server (FastAPI + uvicorn)
│   ├── main.py                    #   FastAPI app with in-memory state
│   ├── requirements.txt           #   Backend-specific dependencies
│   └── ...                        #   Same modules as api/_lib/
│
├── vercel.json                    # Vercel deployment configuration
├── requirements.txt               # Python dependencies (for Vercel)
├── .env.example                   # Environment variable template
└── .gitignore
```

---

## 🚀 Quick Start

### Option A: Deploy to Vercel (Recommended)

**Prerequisites:** [Node.js](https://nodejs.org/) 18+, a [Vercel account](https://vercel.com), and a [Gemini API key](https://aistudio.google.com/apikey)

```bash
# 1. Clone the repository
git clone https://github.com/Aditya-padale/multi-agent.git
cd multi-agent

# 2. Install Vercel CLI
npm i -g vercel

# 3. Deploy
vercel

# 4. Set your Gemini API key in Vercel Dashboard:
#    → Project Settings → Environment Variables
#    → Add: GEMINI_API_KEY = your-api-key
```

### Option B: Run Locally with Vercel Dev

```bash
# 1. Clone and navigate
git clone https://github.com/Aditya-padale/multi-agent.git
cd multi-agent

# 2. Create .env from template
cp .env.example .env
# Edit .env and add your Gemini API key

# 3. Install Vercel CLI and run local dev server
npm i -g vercel
vercel dev
```

### Option C: Run with FastAPI (Legacy)

```bash
# 1. Clone and navigate
git clone https://github.com/Aditya-padale/multi-agent.git
cd multi-agent/backend

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp ../.env.example ../.env
# Edit .env and add your Gemini API key
export $(grep -v '^#' ../.env | xargs)

# 5. Start the server
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — the app serves the frontend directly.

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | Your Google Gemini API key ([get one here](https://aistudio.google.com/apikey)) |
| `GEMINI_MODEL` | ❌ | `gemini-3.5-flash` | Primary model for agent responses |
| `GEMINI_MODEL_FALLBACKS` | ❌ | `gemini-2.5-flash-lite,gemini-2.0-flash,gemini-1.5-flash` | Comma-separated fallback models for 503 errors |

**On Vercel:** Set these in your project's **Settings → Environment Variables** panel.

**Locally:** Create a `.env` file from `.env.example` or export directly in your shell.

---

## 📡 API Reference

All endpoints are serverless Python functions deployed to `/api/*`.

### `GET /api/health`

Health check.

```json
// Response
{ "ok": true }
```

### `POST /api/upload`

Upload a document file (PDF, CSV, XLSX). Returns parsed text chunks for client-side storage.

```bash
curl -X POST /api/upload \
  -F "file=@research-paper.pdf"
```

```json
// Response
{
  "ok": true,
  "filename": "research-paper.pdf",
  "chunks": [
    { "id": "a1b2c3d4", "source": "research-paper.pdf", "doc_type": "pdf", "text": "..." }
  ],
  "total_chunks": 12
}
```

### `POST /api/upload-url`

Fetch and parse a webpage URL.

```json
// Request
{ "url": "https://example.com/article" }

// Response
{
  "ok": true,
  "url": "https://example.com/article",
  "chunks": [...],
  "total_chunks": 8
}
```

### `POST /api/ask`

Run the full debate pipeline. Accepts document chunks and policy state from the client.

```json
// Request
{
  "question": "What are the key findings?",
  "num_agents": 3,
  "debate_rounds": 3,
  "chunks": [...],
  "policy_state": { "literalist": { "alpha": 1.0, "beta": 1.0, ... }, ... }
}

// Response
{
  "question": "What are the key findings?",
  "participants": ["analyst", "skeptic", "explainer"],
  "debate_rounds": 3,
  "debate": [...],
  "agent_results": [...],
  "judge": {
    "scores": { "analyst": { "grounding": 8, "overall": 8.5, ... }, ... },
    "winner": "analyst",
    "best_communicator": "explainer",
    "rationale": "..."
  },
  "final_answer": "...",
  "policy_state": { ... },
  "leaderboard_after": [...]
}
```

---

## ⚙️ Technical Details

### State Management

The serverless architecture uses **client-side state** — no server-side persistence:

| State | Storage | Scope |
|-------|---------|-------|
| Document chunks | Browser memory (`sessionChunks`) | Per-tab, lost on refresh |
| Source list | Browser memory (`sessionSources`) | Per-tab, lost on refresh |
| RL policy | `localStorage` (`arena_policy_state`) | Per-browser, persists across sessions |
| Debate history | Browser memory (`sessionHistory`) | Per-tab, lost on refresh |

### Model Fallback Chain

If the primary Gemini model returns a 503, the client automatically retries with fallback models:

```
gemini-3.5-flash → gemini-2.5-flash-lite → gemini-2.0-flash → gemini-1.5-flash
```

### Judge Scoring Rubric

| Criteria | Weight | Description |
|----------|:------:|-------------|
| Grounding | Standard | Are claims supported by the provided documents? |
| Completeness | Standard | Does the answer fully address the question? |
| Clarity | Standard | Is the answer well-organized and readable? |
| Engagement | **High** | Did the agent meaningfully engage with other agents' arguments? |
| Persuasiveness | **High** | Did the agent's position improve across debate rounds? |

### Vercel Function Limits

| Endpoint | Max Duration | Notes |
|----------|:------------:|-------|
| `/api/upload` | 30s | File parsing is fast |
| `/api/upload-url` | 30s | Depends on target site response time |
| `/api/ask` | 60s | Multiple Gemini API calls; increase to 300s on Vercel Pro |

> **Note:** Full 3-round debates with 4 agents make up to 14 Gemini API calls. On Vercel's free Hobby plan (60s max), consider using 1-round mode for reliability. Vercel Pro supports up to 300s.

---

## 🛣️ Roadmap

- [ ] **Semantic retrieval** — Replace TF-IDF with Gemini embeddings for better paraphrased query matching
- [ ] **Streaming responses** — Stream debate rounds to the frontend as they complete
- [ ] **Persistent storage** — Add Vercel KV / Upstash Redis for cross-device policy persistence
- [ ] **Multi-session support** — Session-scoped document stores for concurrent users
- [ ] **Contextual bandit** — Extend the RL policy to consider question type when selecting agents
- [ ] **Export debates** — Download debate transcripts as PDF/Markdown

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and test locally with `vercel dev`
4. Commit with a descriptive message: `git commit -m "feat: add streaming responses"`
5. Push and open a Pull Request

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">

**Built with [Google Gemini](https://ai.google.dev/) · Deployed on [Vercel](https://vercel.com)**

*Communicate · Compete · Convince*

</div>
