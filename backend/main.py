import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingestion import DocStore, ingest_file, ingest_webpage
from agents import AGENT_PERSONAS
from verifier import judge, judge_debate, normalized_rewards
from combiner import combine, combine_debate
from policy import BanditPolicy, AGENT_KEYS
from debate import run_debate

app = FastAPI(title="Multi-Document Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single in-memory session for this MVP (one user at a time). For multi-user
# use, key these by a session id from the frontend instead.
store = DocStore()
policy = BanditPolicy()
history: list = []


class UrlIn(BaseModel):
    url: str


class AskIn(BaseModel):
    question: str
    num_agents: int = 3       # how many agents Thompson Sampling picks to compete
    debate_rounds: int = 3    # 1 = single-shot (legacy), 2 = open+rebut, 3 = full debate


@app.get("/api/health")
def health():
    return {"ok": True, "documents": len(store.sources()), "chunks": len(store.chunks)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename
    content = await file.read()
    try:
        ingest_file(store, filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse {filename}: {e}")
    return {"ok": True, "filename": filename, "total_chunks": len(store.chunks), "sources": store.sources()}


@app.post("/api/upload-url")
def upload_url(body: UrlIn):
    try:
        ingest_webpage(store, body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch/parse {body.url}: {e}")
    return {"ok": True, "url": body.url, "total_chunks": len(store.chunks), "sources": store.sources()}


@app.get("/api/sources")
def sources():
    return {"sources": store.sources(), "total_chunks": len(store.chunks)}


@app.post("/api/reset-documents")
def reset_documents():
    store.reset()
    return {"ok": True}


@app.post("/api/reset-policy")
def reset_policy():
    policy.reset()
    return {"ok": True}


@app.get("/api/leaderboard")
def leaderboard():
    return {"leaderboard": policy.leaderboard(), "personas": {k: v["label"] for k, v in AGENT_PERSONAS.items()}}


@app.get("/api/history")
def get_history():
    return {"history": history[-20:]}


@app.post("/api/ask")
def ask(body: AskIn):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if store.is_empty():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    k = max(3, min(body.num_agents, len(AGENT_KEYS)))  # min 3 for debates
    chosen_agents = policy.select_agents(k=k)
    chunks = store.retrieve(question, top_k=8)
    num_rounds = max(1, min(body.debate_rounds, 3))

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
    }
    history.append(record)
    return record


# ---- serve the simple frontend ----
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
