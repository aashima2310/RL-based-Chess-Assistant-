FROM python:3.11-slim

# Stockfish is needed by analyzer.py (installs to /usr/games/stockfish)
RUN apt-get update && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

# combined_network.py does "from RL.chess_env.features import ...", so repo root must be on PYTHONPATH
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
WORKDIR /app

COPY product/backend/requirements.txt product/backend/requirements.txt
RUN pip install --no-cache-dir -r product/backend/requirements.txt

COPY RL RL
COPY product product

# Download model weights at build time (too big for GitHub). Edit YOUR_USERNAME.
RUN mkdir -p product/backend/models && python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/akshay1-1/chessrl-model/resolve/main/value_clean_best.pth', 'product/backend/models/value_clean_best.pt')"

WORKDIR /app/product/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
