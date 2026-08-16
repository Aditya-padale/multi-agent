import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api._lib.ingestion import DocStore, ingest_file, ingest_webpage
from api._lib.agents import AGENT_PERSONAS
from api._lib.verifier import judge, judge_debate, normalized_rewards
from api._lib.combiner import combine, combine_debate
from api._lib.policy import BanditPolicy, AGENT_KEYS
from api._lib.debate import run_debate

app = FastAPI(title="Multi-Document Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory fallback session
store = DocStore()
policy = BanditPolicy()
history: list = []


class UrlIn(BaseModel):
    url: str


class AskIn(BaseModel):
    question: str
    num_agents: int = 3
    debate_rounds: int = 3
    chunks: Optional[List[Dict[str, Any]]] = None
    policy_state: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health():
    return {"ok": True, "documents": len(store.sources()), "chunks": len(store.chunks)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    file_bytes = await file.read()
    try:
        chunks = ingest_file(filename, file_bytes)
        return {
            "ok": True,
            "filename": filename,
            "chunks": chunks,
            "total_chunks": len(chunks),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse {filename}: {e}")


@app.post("/api/upload-url")
@app.post("/api/upload_url")
def upload_url(body: UrlIn):
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        chunks = ingest_webpage(url)
        return {
            "ok": True,
            "url": url,
            "chunks": chunks,
            "total_chunks": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch/parse URL: {e}")


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

    # Reconstruct store & policy from client payload if provided, or fallback to in-memory
    if body.chunks:
        current_store = DocStore.from_chunks_list(body.chunks)
    else:
        current_store = store

    if current_store.is_empty():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    current_policy = BanditPolicy(state=body.policy_state) if body.policy_state else policy

    k = max(3, min(body.num_agents, len(AGENT_KEYS)))
    chosen_agents = current_policy.select_agents(k=k)
    retrieved_chunks = current_store.retrieve(question, top_k=8)
    num_rounds = max(1, min(body.debate_rounds, 3))

    debate_data = run_debate(
        chosen_agents=chosen_agents,
        question=question,
        chunks=retrieved_chunks,
        num_rounds=num_rounds,
    )

    if num_rounds > 1:
        judge_result = judge_debate(question, retrieved_chunks, debate_data)
    else:
        agent_results = debate_data["final_answers"]
        judge_result = judge(question, retrieved_chunks, agent_results)
        if "best_communicator" not in judge_result:
            judge_result["best_communicator"] = judge_result.get("winner")

    rewards = normalized_rewards(judge_result["scores"]) if judge_result["scores"] else {}

    if num_rounds > 1:
        final_answer = combine_debate(question, debate_data, judge_result)
    else:
        final_answer = combine(question, debate_data["final_answers"], judge_result)

    if rewards and judge_result.get("winner"):
        current_policy.update(rewards, judge_result["winner"])

    record = {
        "question": question,
        "participants": chosen_agents,
        "debate_rounds": num_rounds,
        "debate": debate_data["rounds"],
        "agent_results": debate_data["final_answers"],
        "judge": judge_result,
        "rewards": rewards,
        "final_answer": final_answer,
        "leaderboard_after": current_policy.leaderboard(),
        "policy_state": current_policy.to_dict(),
    }
    history.append(record)
    return record


handler = app
