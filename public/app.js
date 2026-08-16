/**
 * ARENA — Multi-Agent Debate System
 * Frontend JavaScript — Serverless/Vercel Edition
 *
 * All state is managed client-side:
 *   - Document chunks: stored in sessionChunks (session memory)
 *   - Policy state: stored in localStorage (persists across sessions)
 *   - History: stored in sessionHistory (session memory)
 *   - Sources: tracked in sessionSources (session memory)
 */

const API = "";

// ---- Client-side state ----
let sessionChunks = [];        // All parsed chunks from uploaded docs
let sessionSources = [];       // List of source names (filenames/URLs)
let sessionHistory = [];       // Debate history for this session
let currentDebateData = null;  // Current debate data for round tab rendering

// Policy state — persisted in localStorage
const POLICY_STORAGE_KEY = "arena_policy_state";
const AGENT_KEYS = ["literalist", "analyst", "skeptic", "explainer"];
const AGENT_LABELS = {
  literalist: "The Literalist",
  analyst: "The Analyst",
  skeptic: "The Skeptic",
  explainer: "The Explainer",
};
const DEFAULT_POLICY = () =>
  Object.fromEntries(
    AGENT_KEYS.map((k) => [k, { alpha: 1.0, beta: 1.0, rounds: 0, wins: 0 }])
  );

function loadPolicyState() {
  try {
    const stored = localStorage.getItem(POLICY_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Ensure all keys exist
      for (const k of AGENT_KEYS) {
        if (!parsed[k]) parsed[k] = { alpha: 1.0, beta: 1.0, rounds: 0, wins: 0 };
      }
      return parsed;
    }
  } catch (e) {
    console.warn("Failed to load policy state:", e);
  }
  return DEFAULT_POLICY();
}

function savePolicyState(state) {
  try {
    localStorage.setItem(POLICY_STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    console.warn("Failed to save policy state:", e);
  }
}

// ---- DOM refs ----
const el = (id) => document.getElementById(id);
const statusEl = el("status");
const sourceList = el("sourceList");
const battleField = el("battleField");
const debateTimeline = el("debateTimeline");
const roundTabs = el("roundTabs");
const roundContent = el("roundContent");
const verdictBox = el("verdictBox");
const verdictRationale = el("verdictRationale");
const verdictBadges = el("verdictBadges");
const verdictScores = el("verdictScores");
const finalAnswerBox = el("finalAnswerBox");
const finalAnswerText = el("finalAnswerText");
const leaderboardEl = el("leaderboard");
const askBtn = el("askBtn");

const ROUND_NAMES = {
  opening: "Opening Statements",
  rebuttal: "Cross-Examination",
  closing: "Closing Arguments",
};

const ROUND_COLORS = {
  1: "#6366F1",
  2: "#F59E0B",
  3: "#EF4444",
};

const SCORE_CRITERIA = ["grounding", "completeness", "clarity", "engagement", "persuasiveness", "overall"];
const SCORE_CRITERIA_SHORT = {
  grounding: "GND",
  completeness: "CMP",
  clarity: "CLR",
  engagement: "ENG",
  persuasiveness: "PRS",
  overall: "OVR",
};

// ---- Status ----
function setStatus(msg, isError = false, isLoading = false) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (isError ? " error" : "") + (isLoading ? " status-loading" : "");
}

// ---- Source list (client-side) ----
function refreshSources() {
  if (!sessionSources.length) {
    sourceList.innerHTML = `<p class="empty-note">No documents yet. Upload to begin.</p>`;
    return;
  }
  sourceList.innerHTML = sessionSources
    .map((s) => `<div class="source-item"><span>${escapeHtml(shorten(s, 34))}</span></div>`)
    .join("");
}

// ---- Leaderboard (client-side) ----
function refreshLeaderboard() {
  const state = loadPolicyState();
  const rows = AGENT_KEYS.map((key) => {
    const s = state[key];
    const trust = s.alpha / (s.alpha + s.beta);
    return {
      agent: key,
      trust_score: trust,
      rounds: s.rounds,
      wins: s.wins,
    };
  });
  rows.sort((a, b) => b.trust_score - a.trust_score);

  leaderboardEl.innerHTML = rows
    .map((row, i) => {
      const pct = Math.round(row.trust_score * 100);
      const label = AGENT_LABELS[row.agent] || row.agent;
      return `
        <div class="lb-row ${i === 0 ? "lb-rank1" : ""}">
          <div class="lb-top">
            <span class="lb-name">${escapeHtml(label)}</span>
            <span class="lb-stat">${pct}% · ${row.wins}/${row.rounds}</span>
          </div>
          <div class="lb-bar-track"><div class="lb-bar-fill" style="width:${pct}%"></div></div>
        </div>`;
    })
    .join("");
}

// ---- File upload ----
async function uploadFiles(files) {
  for (const file of files) {
    setStatus(`uploading ${file.name}...`, false, true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      // Store chunks client-side
      if (data.chunks && data.chunks.length) {
        sessionChunks.push(...data.chunks);
      }
      // Track source
      if (!sessionSources.includes(file.name)) {
        sessionSources.push(file.name);
      }
      setStatus(`added ${file.name} (${data.total_chunks} chunks)`);
    } catch (e) {
      setStatus(`failed: ${file.name} — ${e.message}`, true);
    }
  }
  refreshSources();
}

el("fileInput").addEventListener("change", (e) => uploadFiles(e.target.files));

const dropzone = el("dropzone");
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

// ---- URL upload ----
el("urlBtn").addEventListener("click", async () => {
  const url = el("urlInput").value.trim();
  if (!url) return;
  setStatus(`fetching ${url}...`, false, true);
  try {
    const res = await fetch(`${API}/api/upload_url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    // Store chunks client-side
    if (data.chunks && data.chunks.length) {
      sessionChunks.push(...data.chunks);
    }
    // Track source
    if (!sessionSources.includes(url)) {
      sessionSources.push(url);
    }
    setStatus(`added ${url} (${data.total_chunks} chunks)`);
    el("urlInput").value = "";
  } catch (e) {
    setStatus(`failed to fetch url — ${e.message}`, true);
  }
  refreshSources();
});

// ---- Reset controls ----
el("resetDocsBtn").addEventListener("click", () => {
  sessionChunks = [];
  sessionSources = [];
  refreshSources();
  setStatus("evidence cleared");
});

el("resetPolicyBtn").addEventListener("click", () => {
  savePolicyState(DEFAULT_POLICY());
  refreshLeaderboard();
  setStatus("policy reset — agents start fresh");
});

// ---- Ask / Debate ----
askBtn.addEventListener("click", runBattle);
el("questionInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runBattle();
});

async function runBattle() {
  const question = el("questionInput").value.trim();
  const numAgents = parseInt(el("numAgents").value, 10);
  const debateRounds = parseInt(el("debateRounds").value, 10);
  if (!question) return;

  if (!sessionChunks.length) {
    setStatus("No documents uploaded yet. Upload evidence first.", true);
    return;
  }

  askBtn.disabled = true;
  battleField.innerHTML = "";
  debateTimeline.style.display = "none";
  roundTabs.innerHTML = "";
  roundContent.innerHTML = "";
  verdictBox.style.display = "none";
  finalAnswerBox.style.display = "none";
  currentDebateData = null;

  const roundLabel = debateRounds > 1 ? `${debateRounds}-round debate` : "single round";
  setStatus(`${numAgents} agents entering the arena for a ${roundLabel}…`, false, true);

  try {
    const policyState = loadPolicyState();

    const res = await fetch(`${API}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        num_agents: numAgents,
        debate_rounds: debateRounds,
        chunks: sessionChunks,
        policy_state: policyState,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    currentDebateData = data;

    // Update policy state from server response
    if (data.policy_state) {
      savePolicyState(data.policy_state);
    }

    if (data.debate && data.debate.length > 1) {
      renderDebate(data);
    } else {
      renderBattle(data);
    }

    const roundCount = data.debate ? data.debate.length : 1;
    setStatus(`debate complete — ${data.participants.length} agents, ${roundCount} round(s)`);
    refreshLeaderboard();

    // Store in session history
    sessionHistory.push(data);
  } catch (e) {
    setStatus(`error: ${e.message}`, true);
  } finally {
    askBtn.disabled = false;
  }
}

// ---- Render functions (unchanged from original) ----

function renderDebate(data) {
  const rounds = data.debate;
  const winner = data.judge.winner;
  const bestComm = data.judge.best_communicator;

  // Build round tabs
  debateTimeline.style.display = "block";
  roundTabs.innerHTML = rounds
    .map(
      (r, i) =>
        `<button class="round-tab ${i === rounds.length - 1 ? "active" : ""}" data-round="${r.round_number}" onclick="showRound(${i})">${ROUND_NAMES[r.round_name] || r.round_name}</button>`
    )
    .join("");

  // Show the last (most interesting) round by default
  showRound(rounds.length - 1);

  // Render verdict
  renderVerdict(data);

  // Render final answer
  finalAnswerText.textContent = data.final_answer;
  finalAnswerBox.style.display = "block";
}

// Expose showRound globally for onclick
window.showRound = function (roundIndex) {
  if (!currentDebateData) return;
  const rounds = currentDebateData.debate;
  const round = rounds[roundIndex];
  const winner = currentDebateData.judge.winner;
  const bestComm = currentDebateData.judge.best_communicator;
  const scores = currentDebateData.judge.scores || {};

  // Update active tab
  document.querySelectorAll(".round-tab").forEach((tab, i) => {
    tab.classList.toggle("active", i === roundIndex);
  });

  // Render the round's agent cards
  const roundNum = round.round_number;
  const isLastRound = roundIndex === rounds.length - 1;

  roundContent.innerHTML = `
    <div class="round-section">
      <div class="round-label" data-round="${roundNum}">
        <span class="round-indicator r${roundNum}"></span>
        Round ${roundNum} — ${ROUND_NAMES[round.round_name] || round.round_name}
      </div>
      <div class="round-agents">
        ${round.responses
          .map((r) => {
            const isWinner = r.agent === winner && isLastRound;
            const isBestComm = r.agent === bestComm && isLastRound;
            const s = scores[r.agent];
            let scoreLine = "";
            if (s && isLastRound) {
              scoreLine = Object.entries(SCORE_CRITERIA_SHORT)
                .map(([key, abbr]) => `${abbr} ${s[key] ?? "-"}`)
                .join(" · ");
            }
            const body = r.answer
              ? highlightReferences(escapeHtml(r.answer), round.responses)
              : `<em>error: ${escapeHtml(r.error || "no response")}</em>`;
            const classes = [
              "agent-card",
              isWinner ? "winner" : "",
              isBestComm && !isWinner ? "best-communicator" : "",
              isWinner && isBestComm ? "winner best-communicator" : "",
            ]
              .filter(Boolean)
              .join(" ");

            return `
              <div class="${classes}">
                ${isWinner ? `<div class="stamp">WINNER</div>` : ""}
                ${isBestComm ? `<div class="stamp-communicator">BEST COMM</div>` : ""}
                <div class="agent-label">
                  <span class="round-indicator r${roundNum}"></span>
                  ${escapeHtml(r.label)}
                </div>
                ${scoreLine ? `<div class="agent-score">${scoreLine}</div>` : ""}
                <div class="agent-answer">${body}</div>
              </div>`;
          })
          .join("")}
      </div>
    </div>`;
};

function renderVerdict(data) {
  const winner = data.judge.winner;
  const bestComm = data.judge.best_communicator;
  const scores = data.judge.scores || {};
  const personas = {};
  (data.agent_results || []).forEach((r) => {
    personas[r.agent] = r.label;
  });

  // Badges
  let badgesHtml = "";
  if (winner) {
    badgesHtml += `<span class="badge badge-winner">★ ${escapeHtml(personas[winner] || winner)}</span>`;
  }
  if (bestComm && bestComm !== winner) {
    badgesHtml += `<span class="badge badge-communicator">◆ ${escapeHtml(personas[bestComm] || bestComm)}</span>`;
  } else if (bestComm && bestComm === winner) {
    badgesHtml += `<span class="badge badge-communicator">◆ also best communicator</span>`;
  }
  verdictBadges.innerHTML = badgesHtml;

  // Rationale
  verdictRationale.textContent = data.judge.rationale || "";

  // Detailed scores
  const agentKeys = Object.keys(scores);
  verdictScores.innerHTML = agentKeys
    .map((key) => {
      const s = scores[key];
      const label = personas[key] || key;
      const criteria = Object.keys(s).filter((k) => SCORE_CRITERIA.includes(k));
      return `
        <div class="verdict-agent">
          <div class="verdict-agent-name">${escapeHtml(label)}</div>
          ${criteria
            .map((c) => {
              const val = s[c] ?? 0;
              const pct = Math.round((val / 10) * 100);
              return `
                <div class="score-row">
                  <span class="score-label">${SCORE_CRITERIA_SHORT[c] || c}</span>
                  <div class="score-bar-track"><div class="score-bar-fill ${c}" style="width:${pct}%"></div></div>
                  <span class="score-value">${val}</span>
                </div>`;
            })
            .join("")}
        </div>`;
    })
    .join("");

  verdictBox.style.display = "block";
}

function renderBattle(data) {
  // Legacy single-round display
  const winner = data.judge.winner;
  const bestComm = data.judge.best_communicator;
  battleField.innerHTML = (data.agent_results || [])
    .map((r) => {
      const s = (data.judge.scores || {})[r.agent];
      const isWinner = r.agent === winner;
      const isBestComm = r.agent === bestComm;
      const scoreLine = s
        ? `grounding ${s.grounding} · completeness ${s.completeness} · clarity ${s.clarity} · overall ${s.overall}/10`
        : "no score";
      const body = r.answer ? escapeHtml(r.answer) : `<em>error: ${escapeHtml(r.error || "no response")}</em>`;
      return `
        <div class="agent-card ${isWinner ? "winner" : ""} ${isBestComm && !isWinner ? "best-communicator" : ""}">
          ${isWinner ? `<div class="stamp">WINNER</div>` : ""}
          <div class="agent-label">${escapeHtml(r.label)}</div>
          <div class="agent-score">${scoreLine}</div>
          <div class="agent-answer">${body}</div>
        </div>`;
    })
    .join("");

  renderVerdict(data);

  finalAnswerText.textContent = data.final_answer;
  finalAnswerBox.style.display = "block";
}

function highlightReferences(text, allResponses) {
  // Highlight when an agent references another agent by name
  const agentLabels = allResponses.map((r) => r.label).filter(Boolean);
  let result = text;
  for (const label of agentLabels) {
    const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    result = result.replace(
      new RegExp(`\\b(${escaped}|the ${escaped.replace("The ", "")})\\b`, "gi"),
      `<span class="agent-reference">$1</span>`
    );
  }
  return result;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
function shorten(str, n) {
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

// ---- Initial load ----
refreshSources();
refreshLeaderboard();
