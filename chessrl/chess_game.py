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
