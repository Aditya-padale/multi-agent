import os

def _load_env_file():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
        ".env",
    ]
    for path in possible_paths:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip('"').strip("'")
                            if key and key not in os.environ:
                                os.environ[key] = val
            except Exception:
                pass

_load_env_file()

# NEVER hardcode API keys. Set this in your shell or a .env file
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

# Where the RL policy (agent trust scores) persists across restarts
POLICY_STATE_PATH = os.environ.get(
    "POLICY_STATE_PATH",
    os.path.join(os.path.dirname(__file__), "policy_state.json"),
)

if not GEMINI_API_KEY:
    print(
        "[config] WARNING: GEMINI_API_KEY is not set. "
        "Set it as an environment variable before starting the server."
    )
