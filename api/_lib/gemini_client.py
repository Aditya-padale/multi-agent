"""
Minimal wrapper around Google's Gemini generateContent REST endpoint.
No SDK dependency needed -- just requests, so it's easy to swap providers later.
"""
import json
import requests
from api._lib.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MODEL_FALLBACKS, GEMINI_API_BASE


class GeminiError(Exception):
    pass


def generate(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    model: str | None = None,
) -> str:
    """
    Calls Gemini's generateContent endpoint and returns the text of the first candidate.
    Raises GeminiError on failure (missing key, HTTP error, empty response, etc).
    """
    if not GEMINI_API_KEY:
        raise GeminiError(
            "GEMINI_API_KEY is not set. Export it before starting the server."
        )

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
        },
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    model_candidates = []
    for candidate in [model or GEMINI_MODEL, *GEMINI_MODEL_FALLBACKS]:
        if candidate not in model_candidates:
            model_candidates.append(candidate)

    last_error = None
    for model_name in model_candidates:
        url = f"{GEMINI_API_BASE}/{model_name}:generateContent?key={GEMINI_API_KEY}"

        try:
            resp = requests.post(url, json=body, timeout=60)
        except requests.RequestException as e:
            raise GeminiError(f"Network error calling Gemini: {e}") from e

        if resp.status_code == 503 and model_name != model_candidates[-1]:
            last_error = f"Gemini API error {resp.status_code} for {model_name}: {resp.text[:500]}"
            continue

        if resp.status_code != 200:
            raise GeminiError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            candidates = data["candidates"]
            parts = candidates[0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as e:
            raise GeminiError(f"Unexpected Gemini response shape: {json.dumps(data)[:500]}") from e

        if not text.strip():
            raise GeminiError("Gemini returned an empty response (possibly safety-blocked).")

        return text

    raise GeminiError(last_error or "Gemini returned 503 on all configured models.")
