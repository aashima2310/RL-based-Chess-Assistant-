class Config:
    num_simulations = 100
    c_puct = 1.5

    sim_schedule = {
        0:  100,
        5:  200,
        15: 400,
        25: 800
    }

    num_episodes = 100
    temp_threshold = 10
    batch_size = 64
    lr = 0.001
    epochs = 10
    arena_games = 20
    update_threshold = 0.55

    input_size = 40960         
    policy_output_size = 4672  

    checkpoint_dir = "checkpoints/"
    log_path = "logs/training_log.csv"
    buffer_path = "buffer.pkl"
