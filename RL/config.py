class Config:
    num_simulations = 200
    c_puct = 1.5

    sim_schedule = {
        0:  200,
        10:  400,
        20: 800
    }

    num_episodes = 10
    temp_threshold = 20
    batch_size = 256
    lr = 0.003
    epochs = 20
    arena_games = 20
    update_threshold = 0.55

    input_size = 40960         
    policy_output_size = 4672  

    checkpoint_dir = "checkpoints/"
    log_path = "logs/training_log.csv"
    buffer_path = "buffer.pkl"
    num_iterations=200
