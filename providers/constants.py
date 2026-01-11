import os

PROVIDER_CONFIGS = {
    "openai": {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        "api_key": os.environ.get("OPENAI_API_KEY"),
    },
    "openrouter": {
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview"),
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "site_url": os.environ.get("OPENROUTER_SITE_URL"),
        "site_name": os.environ.get("OPENROUTER_SITE_NAME"),
    },
    "ollama": {
        "model": os.environ.get("OLLAMA_MODEL", "gemma3:12b")
    },
}
