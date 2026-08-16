"""
Handles turning uploaded PDFs / CSVs / Excel files / webpages into searchable
text chunks, and retrieving the most relevant chunks for a given question.

Retrieval uses TF-IDF + cosine similarity (scikit-learn). That's a deliberate
choice: it needs no external embedding API, no vector DB, and no GPU -- it
runs instantly and is plenty good for a research-agent demo.

Adapted for serverless: DocStore can be reconstructed from client-sent chunk
dicts, and serialized back to dicts for client storage.
"""
import io
import re
import uuid
from dataclasses import dataclass, field

import pdfplumber
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: str
    source: str          # filename or URL
    doc_type: str         # pdf | csv | xlsx | webpage
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "doc_type": self.doc_type, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(id=d["id"], source=d["source"], doc_type=d["doc_type"], text=d["text"])


@dataclass
class DocStore:
    chunks: list = field(default_factory=list)
    _vectorizer: TfidfVectorizer = None
    _matrix = None

    def add_document(self, source: str, doc_type: str, text: str, chunk_size: int = 900, overlap: int = 150):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        new_chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end]
            chunk = Chunk(id=str(uuid.uuid4())[:8], source=source, doc_type=doc_type, text=piece)
            self.chunks.append(chunk)
            new_chunks.append(chunk)
            start += chunk_size - overlap
        self._rebuild_index()
        return new_chunks

    def _rebuild_index(self):
        if not self.chunks:
            return
        corpus = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 8) -> list:
        if not self.chunks or self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def reset(self):
        self.chunks = []
        self._vectorizer = None
        self._matrix = None

    def sources(self) -> list:
        seen = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    def to_chunks_list(self) -> list:
        """Serialize all chunks to a list of dicts for client storage."""
        return [c.to_dict() for c in self.chunks]

    @classmethod
    def from_chunks_list(cls, chunks_data: list) -> "DocStore":
        """Reconstruct a DocStore from client-sent chunk dicts."""
        store = cls()
        store.chunks = [Chunk.from_dict(d) for d in chunks_data]
        store._rebuild_index()
        return store


# ---------- extractors ----------

def extract_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return _dataframe_to_text(df)


def extract_excel(file_bytes: bytes) -> str:
    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    parts = []
    for name, df in sheets.items():
        parts.append(f"[Sheet: {name}]\n{_dataframe_to_text(df)}")
    return "\n\n".join(parts)


def _dataframe_to_text(df: pd.DataFrame, max_rows: int = 500) -> str:
    df = df.head(max_rows)
    cols = ", ".join(str(c) for c in df.columns)
    lines = [f"Columns: {cols}"]
    for _, row in df.iterrows():
        lines.append(" | ".join(f"{c}={row[c]}" for c in df.columns))
    # Lightweight numeric summary so agents can reason about the data without
    # re-deriving stats from hundreds of raw rows.
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        lines.append("\n[Summary statistics]")
        lines.append(numeric.describe().to_string())
    return "\n".join(lines)


def extract_webpage(url: str) -> str:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def ingest_file(filename: str, file_bytes: bytes) -> list:
    """Parse a file and return chunks as a list of dicts (no server-side storage)."""
    store = DocStore()
    lower = filename.lower()
    if lower.endswith(".pdf"):
        text = extract_pdf(file_bytes)
        store.add_document(filename, "pdf", text)
    elif lower.endswith(".csv"):
        text = extract_csv(file_bytes)
        store.add_document(filename, "csv", text)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        text = extract_excel(file_bytes)
        store.add_document(filename, "xlsx", text)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return store.to_chunks_list()


def ingest_webpage(url: str) -> list:
    """Fetch and parse a webpage, return chunks as a list of dicts."""
    store = DocStore()
    text = extract_webpage(url)
    store.add_document(url, "webpage", text)
    return store.to_chunks_list()
