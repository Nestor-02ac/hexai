"""
MCTS for the Game of Y using UCT + optional RAVE.

Adapted from mcts_hex.py:
- Uses YBoard instead of HexBoard
- Replaces bridge rollouts with connectivity-based heuristic
"""

import math
import random
from y_board import YBoard, Player


class MCTSNode:
    __slots__ = [
        'move', 'player', 'parent', 'children',
        'visits', 'wins',
        'rave_visits', 'rave_wins',
        'untried_moves',
    ]

    def __init__(self, move=None, player=None, parent=None, untried_moves=None):
        self.move = move
        self.player = player
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.rave_visits = {}
        self.rave_wins = {}
        self.untried_moves = untried_moves if untried_moves is not None else []

class SimulationType:

    RANDOM = 1       # completely random playouts
    BRIDGES = 2      # respond to bridge-breaking moves during playouts


class MCTSY:
    """
    Monte-Carlo Tree Search player for Y.
    Simplified from Hex version (no bridges).
    """

    def __init__(self, board_size=5, c_uct=0.5,
                 use_rave=True, rave_bias=0.00025,
                 num_simulations=10000):
        self.board_size = board_size
        self.c_uct = c_uct
        self.use_rave = use_rave
        self.rave_bias = rave_bias
        self.num_simulations = num_simulations

    def select_move(self, board, player):
        empty = board.get_empty_cells()
        if len(empty) == 1:
            return empty[0]

        player_int = int(player)
        opp_int = 3 - player_int

        root = MCTSNode(
            player=opp_int,
            untried_moves=list(empty)
        )

        log = math.log

        for _ in range(self.num_simulations):
            node = root
            sim_board = board.clone()
            cur = player_int

            tree_black_moves = set()
            tree_white_moves = set()

            # Selection
            while not node.untried_moves and node.children:
                best_val = -1.0
                best_child = None
                parent_visits = node.visits

                log_pv = log(parent_visits) if parent_visits > 0 else 0

                for child in node.children:
                    cv = child.visits
                    if cv == 0:
                        val = float('inf')
                    else:
                        val = child.wins / cv
                        if self.c_uct > 0:
                            val += self.c_uct * math.sqrt(log_pv / cv)

                    if val > best_val:
                        best_val = val
                        best_child = child

                node = best_child
                sim_board.play(node.move, cur)

                if cur == 1:
                    tree_black_moves.add(node.move)
                else:
                    tree_white_moves.add(node.move)

                cur = 3 - cur

            # Expansion
            if node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                move = node.untried_moves[idx]
                node.untried_moves[idx] = node.untried_moves[-1]
                node.untried_moves.pop()

                sim_board.play(move, cur)

                if cur == 1:
                    tree_black_moves.add(move)
                else:
                    tree_white_moves.add(move)

                child = MCTSNode(
                    move=move,
                    player=cur,
                    parent=node,
                    untried_moves=sim_board.get_empty_cells()
                )
                node.children.append(child)
                node = child
                cur = 3 - cur

            # Simulation (pure random)
            empties = sim_board.get_empty_cells()
            random.shuffle(empties)

            black_sim = set(tree_black_moves)
            white_sim = set(tree_white_moves)

            p = cur
            for cell in empties:
                sim_board.play(cell, p)
                if p == 1:
                    black_sim.add(cell)
                else:
                    white_sim.add(cell)
                p = 3 - p

            # Winner
            if sim_board.check_win(1):
                winner = 1
            else:
                winner = 2

            # Backpropagation
            current = node
            while current is not None:
                current.visits += 1
                if current.player == winner:
                    current.wins += 1.0

                current = current.parent

        if not root.children:
            return random.choice(empty)

        best = max(root.children, key=lambda c: c.visits)
        return best.move


class RandomAgent:
    def select_move(self, board, player):
        return random.choice(board.get_empty_cells())
    

if __name__ == "__main__":
    board = YBoard(size=5)
    mcts_agent = MCTSY(board_size=5, c_uct=1.4, rave_bias=0.00025, use_rave=True, num_simulations=10000)
    random_agent = RandomAgent()

    current_player = Player.BLACK
    while True:
        if current_player == Player.BLACK:
            move = mcts_agent.select_move(board, current_player)
        else:
            move = random_agent.select_move(board, current_player)

        board.play(move, current_player)
        board.display()
        print()

        if board.check_win(current_player):
            print(f"{current_player} wins!")
            break

        current_player = Player.WHITE if current_player == Player.BLACK else Player.BLACK