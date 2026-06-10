# ChessRL — Product Side

## Setup (do this once)

```bash
cd product
pip install -r requirements.txt

# Install Stockfish (used as placeholder until RL engine is ready)
# Ubuntu/WSL:
sudo apt install stockfish
# Mac:
brew install stockfish
# Windows: download from https://stockfishchess.org/download/
```

## Download Puzzle Database (do this once, ~250MB)

```bash
# Download from Lichess — pick the latest
wget https://database.lichess.org/lichess_db_puzzle.csv.zst -O data/puzzles.zst
# Decompress (install zstd first: sudo apt install zstd)
zstd -d data/puzzles.zst -o data/lichess_db_puzzle.csv
```

## Run (two terminals)

**Terminal 1 — Backend:**
```bash
cd product
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd product
streamlit run app.py
```

Open browser at: http://localhost:8501

## Swap to RL Engine (when team is ready)

In `backend/analyzer.py`, line 18:
```python
# Change this:
ENGINE_PATH = "stockfish"
# To this:
ENGINE_PATH = "path/to/chessrl_engine"  # UCI compliant binary from RL team
```

That's the only change needed.

## File Structure

```
product/
  app.py                  ← Streamlit UI (all 4 pages)
  requirements.txt
  backend/
    main.py               ← FastAPI endpoints
    analyzer.py           ← PGN analysis (Pipeline 2A)
    puzzle.py             ← Puzzle recommender (Pipeline 3)
    chatbot.py            ← LLM chatbot (Pipeline 4)
    database.py           ← SQLite player profiles
  data/
    lichess_db_puzzle.csv ← Download separately (see above)
    chessrl.db            ← Auto-created on first run
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload_pgn | Analyze a PGN game |
| GET | /puzzles/{user_id} | Get puzzle recommendations |
| POST | /chat | Ask chatbot about a game |
| GET | /profile/{user_id} | Get player profile |
| GET | /analysis/{game_id} | Get stored analysis |
| GET | /health | Check if backend is running |
