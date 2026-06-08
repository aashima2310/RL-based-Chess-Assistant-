import chess
import chess.engine
import random
import pickle
from tqdm import tqdm
def generate_random_game_positions():
  positions=[]
  board=chess.Board()
  while not board.is_game_over():
    legal_moves=list(board.legal_moves)
    random_move=random.choice(legal_moves)
    board.push(random_move)
    positions.append(board.fen())
  return positions
def collect_data(target=500000,stockfish_path="/usr/games/stockfish",depth=10):
   data=[]
   with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as engine:
    with tqdm(total=target) as pbar:
      while len(data)<target:
        positions=generate_random_game_positions()
        for fen in positions:
          if len(data)>=target:
            break
          board=chess.Board(fen)
          if board.is_game_over():
            continue
          info=engine.analyse(board,chess.engine.Limit(depth=depth))
          score=info["score"].white()
          if score.is_mate():
            cp=10000 if score.mate()>0 else -10000
          else:
            cp=score.score(default=0)
          data.append((fen,cp))
          pbar.update(1)
   return data
if __name__=='__main__':
  print("Collecting 500000 positions")
  data= collect_data()
  with open("stockfish_data.pkl",'wb') as f:
    pickle.dump(data,f)
