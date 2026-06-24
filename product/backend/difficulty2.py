DIFFICULTY_SIMULATIONS = {
    "easy": 50,
    "medium": 200,
    "hard": 800
}


def get_simulations(difficulty):
    return DIFFICULTY_SIMULATIONS.get(difficulty.lower(), 50)