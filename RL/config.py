class Config:
    num_simulations       = 200
    c_puct                 = 1.5
    sim_schedule           = {0: 200}  
    dirichlet_alpha        = 0.3        
    dirichlet_epsilon      = 0.25

    @classmethod
    def get_num_simulations(cls, iteration):
        
        applicable = [k for k in cls.sim_schedule if k <= iteration]
        key = max(applicable) if applicable else 0
        return cls.sim_schedule[key]

    unfreeze_at_iteration  = 100
    num_episodes           = 6
    temp_threshold          = 10
    batch_size              = 256
    lr                      = 0.003
    epochs                  = 20
    arena_games             = 20
    update_threshold        = 0.55
    input_size              = 40960
    policy_output_size      = 4672
    checkpoint_dir          = "checkpoints/"
    log_path                = "logs/training_log.csv"
    buffer_path             = "buffer.pkl"
    num_iterations          = 200

    BOOK_PATH    = "/content/drive/MyDrive/chess_rl/opening_book.bin"
    MAX_BOOK_PLY = 20

    SYZYGY_PATH       = "/content/drive/MyDrive/chess_rl/syzygy"
    SYZYGY_MAX_PIECES = 5
    ELITE_WEIGHT  = 0.5
    PUZZLE_WEIGHT = 0.3
    RANDOM_WEIGHT = 0.2

    STOCKFISH_PATH      = "/usr/games/stockfish"
    STOCKFISH_DEPTH     = 14
    STOCKFISH_MULTIPV   = 8
    DISTILL_TEMPERATURE = 0.4
    POLICY_TOP1_GATE   = 0.30
    POLICY_TOP3_GATE   = 0.55
    VALUE_PEARSON_GATE = 0.6  
