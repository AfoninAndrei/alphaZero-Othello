# Dockerfile  ─ place at repo root
FROM python:3.10-slim

# ─── System libs needed for pygame/SDL ────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libx11-6 libxext6 libxrender-dev libxtst6 && \
    rm -rf /var/lib/apt/lists/*

# ─── Python deps (CPU-only) ───────────────────────────────────────────
RUN pip install --no-cache-dir \
        torch==2.4.0 \
        pygame numpy

# ─── Copy source + model(s) ───────────────────────────────────────────
WORKDIR /app
COPY play_othello.py MCTS_model.py Models.py .
COPY othello_policy_RL_small.pt othello_policy_RL_big.pt .

COPY envs/ envs/
COPY ui/ ui/

# ─── Default action: launch the UI game ───────────────────────────────
CMD ["python", "play_othello.py"]
