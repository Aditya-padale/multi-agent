"""
Configuration for Vercel serverless deployment.
All config is read from environment variables (set in Vercel dashboard).
No .env file loading — Vercel handles environment variable injection.
"""
import os

# NEVER hardcode API keys. Set GEMINI_API_KEY in Vercel dashboard.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Use an explicit model so the app doesn't ride a busy alias during traffic spikes.
# Override with GEMINI_MODEL if you want to pin a different model.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# Ordered fallback models to try if Gemini returns a temporary 503/UNAVAILABLE.
# You can override this with a comma-separated GEMINI_MODEL_FALLBACKS list.
GEMINI_MODEL_FALLBACKS = tuple(
    model.strip()
    for model in os.environ.get(
        "GEMINI_MODEL_FALLBACKS",
        "gemini-2.5-flash-lite,gemini-2.0-flash,gemini-1.5-flash",
    ).split(",")
    if model.strip()
)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

if not GEMINI_API_KEY:
    print(
        "[config] WARNING: GEMINI_API_KEY is not set. "
        "Set it in your Vercel dashboard environment variables."
    )
