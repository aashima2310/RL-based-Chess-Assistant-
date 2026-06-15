import numpy as np
import math
import chess
import torch
import sys
sys.path.append('/content/RL-based-Chess-Assistant-')
from RL.chess_env.features import HalfKPExtractor
halfkp_extractor = HalfKPExtractor()
class Chess_game:
  def __init__(self):
   self.row_count=8
   self.column_count=8
   self.action_size = 4672 

  def get_initial_state(self):
    return chess.Board()

def get_next_state(self, state, action):
    state = state.copy()
    move = halfkp_extractor.idx_to_move(action, state)
    if move in state.legal_moves:
        state.push(move)
    else:
        promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        if promo_move in state.legal_moves:
            state.push(promo_move)
        else:
            legal = list(state.legal_moves)
            if legal:
                state.push(legal[0])
    return state

  def get_valid_moves(self,state):
    mask = torch.zeros(4672, dtype=torch.float32)
    for move in state.legal_moves:
        if move.promotion is None:
            piece = state.piece_at(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            if piece and piece.piece_type == chess.PAWN and to_rank in (0, 7):
                move = chess.Move(move.from_square, move.to_square,
                                  promotion=chess.QUEEN)
        try:
            idx = halfkp_extractor.move_to_idx(move)
            mask[idx] = 1.0
        except (ValueError, IndexError):
            pass   
    return mask

  def get_value_and_terminated(self, state, action):
    if state.is_checkmate():
        return 1, True 
    if state.is_stalemate() or state.is_insufficient_material() or state.is_seventyfive_moves() or state.is_fivefold_repetition():
        return 0, True 
    return 0, False 

  def get_opponent(self,player):
    if isinstance(player, str):
        if player=="WHITE":
            return "BLACK"
        elif player=="BLACK":
            return "WHITE"
    elif isinstance(player, int):
        if player == 1:
            return -1
        elif player == -1:
            return 1
    return None
  
  def get_opponent_value(self, value):
    return -value

  def change_perspective(self,state):
    return state.mirror()

