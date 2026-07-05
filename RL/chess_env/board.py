import chess
import sys
sys.path.append('/content/RL-based-Chess-Assistant-')
from RL.chess_env.features import HalfKPExtractor, resolve_move

halfkp_extractor = HalfKPExtractor()


class Chess_game:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 4672

    def get_initial_state(self):
        return chess.Board()

    def get_next_state(self, state, action):

        new_state, _ = self.get_next_state_and_action(state, action)
        return new_state

    def get_next_state_and_action(self, state, action, policy=None):
        state = state.copy()
        move, true_action = resolve_move(action, state, policy)
        state.push(move)
        return state, true_action

    def get_valid_moves(self, state):
        return halfkp_extractor.get_legal_moves(state)

    def get_value_and_terminated(self, state, action):

        if state.is_checkmate():
            return -1, True
        if (state.is_stalemate() or state.is_insufficient_material()
                or state.is_seventyfive_moves() or state.is_fivefold_repetition()):
            return 0, True
        return 0, False

    def get_opponent(self, player):
        if isinstance(player, str):
            if player == "WHITE":
                return "BLACK"
            elif player == "BLACK":
                return "WHITE"
        elif isinstance(player, int):
            if player == 1:
                return -1
            elif player == -1:
                return 1
        return None

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state):
        return state.mirror()
