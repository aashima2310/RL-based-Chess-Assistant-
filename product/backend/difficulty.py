DIFFICULTY_SIMULATIONS = {
    "easy": 300,
    "medium": 700,
    "hard": 1600
}

def get_simulations(difficulty):
    return DIFFICULTY_SIMULATIONS.get(difficulty.lower(), 50)
