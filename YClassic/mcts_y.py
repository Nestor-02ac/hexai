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

            # Selection with RAVE
            while not node.untried_moves and node.children:
                best_val = -1.0
                best_child = None
                parent_visits = node.visits

                log_pv = log(parent_visits) if parent_visits > 0 else 0

                if self.use_rave:
                    n_rv = node.rave_visits
                    n_rw = node.rave_wins

                    for child in node.children:
                        cv = child.visits

                        if cv == 0:
                            rc = n_rv.get(child.move, 0)
                            if rc > 0:
                                val = n_rw.get(child.move, 0) / rc
                            else:
                                val = float('inf')
                        else:
                            m = child.wins / cv
                            rc = n_rv.get(child.move, 0)

                            if rc > 0:
                                rw = n_rw.get(child.move, 0)
                                coef = 1.0 - rc / (rc + cv + rc * cv * self.rave_bias)

                                # clamp for safety
                                if coef < 0.0:
                                    coef = 0.0
                                elif coef > 1.0:
                                    coef = 1.0

                                val = m * coef + (1.0 - coef) * (rw / rc)
                            else:
                                val = m

                            if self.c_uct > 0 and parent_visits > 0:
                                val += self.c_uct * math.sqrt(log_pv / cv)

                        if val > best_val:
                            best_val = val
                            best_child = child

                else:
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

            # Simulation (improved rollout)
            empties = sim_board.get_empty_cells()

            black_sim = set(tree_black_moves)
            white_sim = set(tree_white_moves)

            p = cur

            def move_score(cell, player):
                """Heuristic: favor connectivity + side expansion"""
                score = 0

                # 1. Favor neighbors of same color (connect groups)
                for n in sim_board._neighbors[cell]:
                    if sim_board.board[n] == player:
                        score += 2
                    elif sim_board.board[n] == 3 - player:
                        score += 0.5  # mild blocking

                # 2. Favor touching new sides
                if cell in sim_board._side_a:
                    score += 1
                if cell in sim_board._side_b:
                    score += 1
                if cell in sim_board._side_c:
                    score += 1

                return score


            while empties:
                # Pick best move among a random subset
                k = min(6, len(empties))  # sample size (tune 4–10)
                sample = random.sample(empties, k)

                best_cell = max(sample, key=lambda c: move_score(c, p))

                empties.remove(best_cell)

                sim_board.play(best_cell, p)

                if p == 1:
                    black_sim.add(best_cell)
                else:
                    white_sim.add(best_cell)

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

                if self.use_rave and current.player is not None:
                    next_p = 3 - current.player
                    moves_set = black_sim if next_p == 1 else white_sim

                    rv = current.rave_visits
                    rw = current.rave_wins
                    is_win = (winner == next_p)

                    for mv in moves_set:
                        rv[mv] = rv.get(mv, 0) + 1
                        if is_win:
                            rw[mv] = rw.get(mv, 0) + 1.0

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
    