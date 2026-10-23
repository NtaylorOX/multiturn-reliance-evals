"""Short names for the models used in the study (Inspect model strings)."""

MODELS = {
    # Hosted on NVIDIA's free endpoint (needs NVIDIA_API_KEY and NVIDIA_BASE_URL in .env)
    "nemotron": "openai-api/nvidia/nvidia/nemotron-3-ultra-550b-a55b",
    # Served locally by Ollama
    "llama31-8b": "ollama/llama3.1:8b",
    "deepseek-r1": "ollama/deepseek-r1:latest",
}

SIMULATOR = MODELS["nemotron"]
JUDGE = MODELS["nemotron"]
