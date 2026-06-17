import numpy as np
from chess_env.board import Chess_game
from mcts.search import MCTS
from config import Config


def play_match(game, mcts1, mcts2):

    state = game.get_initial_state()
    player = 1

    while True:

        if player == 1:
            neutral_state = state
            action_probs = mcts1.search(neutral_state)
        else:
            neutral_state = game.change_perspective(state)
            action_probs = mcts2.search(neutral_state)
          
        action = np.argmax(action_probs)
        state = game.get_next_state(state, action)
        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            if value == 0:
                return 0

            if player == 1:
                return -1
            else:
                return 1

        player = game.get_opponent(player)


def evaluate(champion_network, challenger_network):

    game = Chess_game()
    sims = Config.num_simulations
    for threshold, count in sorted(Config.sim_schedule.items()):
        if iteration >= threshold:
            sims = count
    args = {
        'C': Config.c_puct,
        'num_searches': sims,        
        'add_noise': False,          
        'device': next(champion_network.parameters()).device
    }

    champion_mcts   = MCTS(game, args)
    challenger_mcts = MCTS(game, args)
  
    champion_mcts.network   = champion_network
    challenger_mcts.network = challenger_network

    challenger_wins = 0
    draws           = 0
    champion_wins   = 0

    for game_num in range(Config.arena_games):

        if game_num % 2 == 0:
            result = play_match(game, champion_mcts, challenger_mcts)
        else:
            result = play_match(game, challenger_mcts, champion_mcts)
            result = -result

        if result == 1:
            challenger_wins += 1
        elif result == -1:
            champion_wins += 1
        else:
            draws += 1

        print(f"Game {game_num+1}: "
              f"Challenger {challenger_wins} | "
              f"Draws {draws} | "
              f"Champion {champion_wins}")

    total_games  = challenger_wins + champion_wins + draws
    win_rate     = challenger_wins / total_games

    print(f"\nChallenger win rate: {win_rate:.2%}")

    if win_rate >= Config.update_threshold:
        print("Challenger wins! Replacing champion.")
        return True

    else:
        print("Champion holds. Keeping current champion.")
        return False
