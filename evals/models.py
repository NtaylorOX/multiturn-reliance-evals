"""Short names for the models used in the study (Inspect model strings)."""

# The models under study (targets).
MODELS = {
    # Hosted on NVIDIA's free endpoint (needs NVIDIA_API_KEY and NVIDIA_BASE_URL in .env)
    "nemotron": "openai-api/nvidia/nvidia/nemotron-3-ultra-550b-a55b",
    # Served locally by Ollama
    "llama31-8b": "ollama/llama3.1:8b",
    "deepseek-r1": "ollama/deepseek-r1:latest",
}

# The user simulator and the judge run locally, so they never wait on a hosted rate limit.
# Qwen3 has a soft switch: "/no_think" in the prompt turns its reasoning off (see evals.conversation.no_think).
SIMULATOR = "ollama/qwen3:8b"
JUDGE = "ollama/qwen3:8b"
