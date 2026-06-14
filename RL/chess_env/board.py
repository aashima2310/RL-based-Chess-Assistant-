import numpy as np
import math
import chess
def active_features(board: chess.Board):

    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    white_active_feats = []
    black_active_feats = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type!=chess.KING:
            piece_type_int = PIECE_TO_INT[piece.symbol()]
            if white_king_sq is not None:
                white_active_feats.append(half_kp(white_king_sq, piece_type_int, sq))
            if black_king_sq is not None:
                black_active_feats.append(half_kp(flip_sq(black_king_sq), piece_type_int, flip_sq(sq)))

    return white_active_feats, black_active_feats
class Chess_game:
  def __init__(self):
   self.row_count=8
   self.column_count=8
   self.action_size=4672
  def get_initial_state(self):
    return chess.Board()
  def get_next_state(self,state,action):
    state=state.copy()
    move=index_to_move(action,state)
    state.push(move)
    return state
  def get_valid_moves(self,state):
    legal=np.zeros(self.action_size,dtype=np.int8)
    for move in state.legal_moves:
        legal[move_to_index(move)] = 1
    return legal
  def get_value_and_terminated(self, state, action):
    if state.is_checkmate():
        return 1, True
    if state.is_game_over():
        return 0, True
    return 0, False
  def get_opponent(self,player):
    if player=="WHITE":
      return "BLACK"
    elif player=="BLACK":
      return "WHITE"
    elif player == 1: 
      return -1
    elif player == -1: 
      return 1
    return None
  def get_opponent_move(self,move):
    return move^56
  def get_opponent_value(self, value):
    return -value
  def change_perspective(self,state):
    return state.mirror()
chess_board=chess.board()
player="WHITE"
arg={'C': cpuct,"num_searches": 1000} 


model = ChessNet(chess_board.action_size)

mcts=MCTS(chess_board, arg, model)
state=chess_board.get_initial_state()

while True:
  print(state)
  valid = chess_board.get_valid_moves(state)

  if player == "WHITE":
   uci = input(f"Enter move for {player} (e.g. e2e4): ")
   try:
            move = chess.Move.from_uci(uci)
            action = halfkp_extractor.move_to_idx(move)
   except ValueError:
            print("Invalid UCI format, try again")
            continue

   if valid[action] == 0:
            print("Illegal move, try again")
            continue
  else:
    
    mcts_probs = mcts.search(state) 
    action = np.argmax(mcts_probs)

    move_obj = halfkp_extractor.idx_to_move(action)
    print(f"MCTS chose move: {move_obj.uci()}")

    if valid[action] == 0:
            print("MCTS chose illegal move, picking random legal move")
            legal_moves_indices = np.where(valid == 1)[0]
            if len(legal_moves_indices) > 0:
                action = np.random.choice(legal_moves_indices)
            else:
                print("No legal moves available, game should be terminated.")
                break

  state=chess_board.get_next_state(state,action)
  value, terminate=chess_board.get_value_and_terminated(state,action)

  if terminate:
    print(state)
    if value==1:
        print(f"Game ended, player {player} won")
    elif value==0:
        print("Game ended in draw")
    break

  player=chess_board.get_opponent(player)
