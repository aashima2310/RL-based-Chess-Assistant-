import numpy as np
import chess
import sys
sys.path.append('/content/RL-based-Chess-Assistant-')

from RL.chess_env.board import Chess_game
from RL.mcts.search import MCTS
from RL.config import Config
from RL.chess_env.features import HalfKPExtractor

halfkp_extractor = HalfKPExtractor()


def play_one_game(game, mcts, network):
    state = game.get_initial_state()
    training_data = []
    player = 1

    while True:
        if player == -1:
            canonical_state = game.change_perspective(state)
        else:
            canonical_state = state.copy()

        action_probs = mcts.search(canonical_state)
        training_data.append((state.copy(), action_probs, player))

        move_number = state.fullmove_number
        if move_number < Config.temp_threshold:
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            action = np.argmax(action_probs)

        if player == -1:
            canonical_move = halfkp_extractor.idx_to_move(action, canonical_state)
            original_move = chess.Move(
                chess.square_mirror(canonical_move.from_square),
                chess.square_mirror(canonical_move.to_square),
                promotion=canonical_move.promotion
            )
            action = halfkp_extractor.move_to_idx(original_move)

        state = game.get_next_state(state, action)
        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            return_data = []
            for hist_state, hist_probs, hist_player in training_data:
                if value == 0:
                    hist_value = 0
                elif hist_player == player:
                    hist_value = value
                else:
                    hist_value = game.get_opponent_value(value)
                return_data.append((hist_state, hist_probs, hist_value))
            return return_data

        player = game.get_opponent(player)


def run_self_play(network, iteration=0, args=None):
    game = Chess_game()
    device = next(network.parameters()).device

    sims = Config.num_simulations
    for threshold, count in sorted(Config.sim_schedule.items()):
        if iteration >= threshold:
            sims = count

    args = {
        'C': Config.c_puct,
        'num_searches': sims,
        'add_noise': True,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'device': device
    }

    mcts = MCTS(game, args, network)
    all_data = []

    for episode in range(Config.num_episodes):
        print(f"Episode {episode + 1}/{Config.num_episodes}")
        game_data = play_one_game(game, mcts, network)
        all_data.extend(game_data)

    return all_data
