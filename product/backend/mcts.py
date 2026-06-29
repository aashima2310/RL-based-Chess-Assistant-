import math
import random
import chess

from difficulty import get_simulations


class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move

        self.children = []
        self.visits = 0
        self.value = 0.0

    def expand(self):
        if self.children:
            return

        for move in self.board.legal_moves:
            child_board = self.board.copy()
            child_board.push(move)

            child_node = MCTSNode(
                board=child_board,
                parent=self,
                move=move
            )

            self.children.append(child_node)

    def average_value(self):
        if self.visits == 0:
            return 0.0

        return self.value / self.visits


class MCTS:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self.simulations = get_simulations(difficulty)

        self.exploration_constant = 1.4
        self.engine_color = chess.WHITE

        print("MCTS started")
        print("Difficulty:", difficulty)
        print("Simulations:", self.simulations)

    def evaluate_board(self, board):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

        score = 0

        for piece_type, value in piece_values.items():
            white_count = len(
                board.pieces(piece_type, chess.WHITE)
            )

            black_count = len(
                board.pieces(piece_type, chess.BLACK)
            )

            score += white_count * value
            score -= black_count * value

        if self.engine_color == chess.WHITE:
            return score

        return -score

    def select_child(self, node):
        unvisited_children = [
            child
            for child in node.children
            if child.visits == 0
        ]

        if unvisited_children:
            return random.choice(unvisited_children)

        best_child = None
        best_score = float("-inf")

        for child in node.children:
            exploitation = child.average_value()

            exploration = self.exploration_constant * math.sqrt(
                math.log(node.visits) / child.visits
            )

            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def backpropagate(self, node, score):
        while node is not None:
            node.visits += 1
            node.value += score
            node = node.parent

    def search(self, board):
        if board.is_game_over():
            print("Game is already over.")
            return None

        self.engine_color = board.turn

        root = MCTSNode(board.copy())
        root.expand()

        print("AI is thinking...")
        print("Root children:", len(root.children))

        for _ in range(self.simulations):
            first_child = self.select_child(root)

            first_child.expand()

            if first_child.children:
                second_child = self.select_child(first_child)
                score = self.evaluate_board(second_child.board)
                self.backpropagate(second_child, score)

            else:
                score = self.evaluate_board(first_child.board)
                self.backpropagate(first_child, score)

        best_child = max(
            root.children,
            key=lambda child: child.visits
        )

        print("Chosen move:", best_child.move)
        print("Visits:", best_child.visits)
        print("Average value:", best_child.average_value())
        print("Root visits:", root.visits)

        return best_child.move