"""
Monte-Carlo Tree Search for Y with UCT + RAVE.

This mirrors the HexClassic Python reference closely:
  - one correct UCT/RAVE selection loop
  - AMAF/RAVE backpropagation
  - playouts that fill the board (Y, like Hex, has no draws on full boards)

Y does not have Hex's bridge templates, so the stronger rollout uses a local
connectivity heuristic based on component side masks instead.
"""

import math
import random

try:
    from y_board import Player, YBoard
except ModuleNotFoundError:
    from .y_board import Player, YBoard


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = [
        "move",
        "player",
        "parent",
        "children",
        "visits",
        "wins",
        "rave_visits",
        "rave_wins",
        "untried_moves",
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
    """Supported rollout policies."""

    RANDOM = 1
    CONNECTIVITY = 2


class MCTSY:
    """
    Monte-Carlo Tree Search player for Y.

    The connectivity rollout favors moves that:
      - merge friendly components,
      - expand the set of sides touched by that component,
      - occupy tactically relevant cells next to strong enemy groups.
    """

    def __init__(
        self,
        board_size=9,
        c_uct=0.0,
        rave_bias=0.00025,
        use_rave=True,
        simulation_type=SimulationType.CONNECTIVITY,
        num_simulations=10000,
        rollout_sample_size=6,
    ):
        self.board_size = board_size
        self.c_uct = c_uct
        self.rave_bias = rave_bias
        self.use_rave = use_rave
        self.simulation_type = simulation_type
        self.num_simulations = num_simulations
        self.rollout_sample_size = rollout_sample_size

    def _rollout_score(self, board, cell, player):
        """
        Score a rollout move using Y-specific connectivity features.

        The board is small enough that a local 6-neighbor analysis is cheap and
        useful, especially because Y's objective is exactly "connect components
        so their side masks accumulate to 0b111".
        """
        own_mask = board._cell_side_mask[cell]
        opp_mask = 0
        own_roots = []
        opp_roots = []
        own_neighbors = 0
        opp_neighbors = 0
        opp = 3 - player

        for nidx in board._neighbors[cell]:
            stone = board.board[nidx]
            if stone == player:
                own_neighbors += 1
                root = board._find(nidx, player)
                if root not in own_roots:
                    own_roots.append(root)
                    own_mask |= board.component_mask[player][root]
            elif stone == opp:
                opp_neighbors += 1
                root = board._find(nidx, opp)
                if root not in opp_roots:
                    opp_roots.append(root)
                    opp_mask |= board.component_mask[opp][root]

        score = 16 * own_mask.bit_count()
        score += 6 * len(own_roots)
        score += 2 * own_neighbors
        score += 4 * board._cell_side_mask[cell].bit_count()
        score += 3 * opp_mask.bit_count()
        score += opp_neighbors

        if own_mask == board.ALL_SIDES:
            score += 1000
        if opp_mask == board.ALL_SIDES:
            score += 120

        return score

    def select_move(self, board, player):
        """Run MCTS and return the best move for player."""
        empty = board.get_empty_cells()
        if len(empty) == 1:
            return empty[0]

        player_int = int(player)
        opp_int = 3 - player_int

        root = MCTSNode(
            player=opp_int,
            untried_moves=list(empty),
        )

        use_rave = self.use_rave
        c_uct = self.c_uct
        rave_bias = self.rave_bias
        use_connectivity = self.simulation_type == SimulationType.CONNECTIVITY
        rollout_sample_size = self.rollout_sample_size
        log = math.log
        sqrt = math.sqrt
        n = board.n

        for _ in range(self.num_simulations):
            node = root
            sim_board = board.clone()
            cur = player_int

            black_seen = bytearray(n)
            white_seen = bytearray(n)
            black_moves = []
            white_moves = []

            # Selection
            while not node.untried_moves and node.children:
                best_val = float("-inf")
                best_child = None
                parent_visits = node.visits

                if use_rave:
                    log_pv = log(parent_visits) if parent_visits > 0 else 0.0
                    node_rave_visits = node.rave_visits
                    node_rave_wins = node.rave_wins
                    for child in node.children:
                        child_visits = child.visits
                        if child_visits == 0:
                            rave_count = node_rave_visits.get(child.move, 0)
                            if rave_count > 0:
                                val = node_rave_wins.get(child.move, 0.0) / rave_count
                            else:
                                val = float("inf")
                        else:
                            mean_value = child.wins / child_visits
                            rave_count = node_rave_visits.get(child.move, 0)
                            if rave_count > 0:
                                rave_value = node_rave_wins.get(child.move, 0.0) / rave_count
                                coef = 1.0 - rave_count / (
                                    rave_count
                                    + child_visits
                                    + rave_count * child_visits * rave_bias
                                )
                                if coef < 0.0:
                                    coef = 0.0
                                elif coef > 1.0:
                                    coef = 1.0
                                val = mean_value * coef + (1.0 - coef) * rave_value
                            else:
                                val = mean_value
                            if c_uct > 0.0 and parent_visits > 0:
                                val += c_uct * sqrt(log_pv / child_visits)
                        if val > best_val:
                            best_val = val
                            best_child = child
                else:
                    log_pv = log(parent_visits) if parent_visits > 0 else 0.0
                    for child in node.children:
                        child_visits = child.visits
                        if child_visits == 0:
                            val = float("inf")
                        else:
                            val = child.wins / child_visits
                            if c_uct > 0.0 and parent_visits > 0:
                                val += c_uct * sqrt(log_pv / child_visits)
                        if val > best_val:
                            best_val = val
                            best_child = child

                node = best_child
                sim_board.play_unchecked(node.move, cur)
                if cur == 1:
                    if not black_seen[node.move]:
                        black_seen[node.move] = 1
                        black_moves.append(node.move)
                else:
                    if not white_seen[node.move]:
                        white_seen[node.move] = 1
                        white_moves.append(node.move)
                cur = 3 - cur

            # Expansion
            if node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                move = node.untried_moves[idx]
                node.untried_moves[idx] = node.untried_moves[-1]
                node.untried_moves.pop()

                sim_board.play_unchecked(move, cur)
                if cur == 1:
                    if not black_seen[move]:
                        black_seen[move] = 1
                        black_moves.append(move)
                else:
                    if not white_seen[move]:
                        white_seen[move] = 1
                        white_moves.append(move)

                child = MCTSNode(
                    move=move,
                    player=cur,
                    parent=node,
                    untried_moves=sim_board.get_empty_cells(),
                )
                node.children.append(child)
                node = child
                cur = 3 - cur

            # Simulation
            empties = sim_board.get_empty_cells()
            p = cur

            while empties:
                if use_connectivity:
                    sample_size = min(rollout_sample_size, len(empties))
                    sample_positions = (
                        range(len(empties))
                        if sample_size == len(empties)
                        else random.sample(range(len(empties)), sample_size)
                    )

                    best_pos = None
                    best_cell = None
                    best_score = None
                    for pos in sample_positions:
                        cell = empties[pos]
                        score = self._rollout_score(sim_board, cell, p)
                        if best_score is None or score > best_score:
                            best_score = score
                            best_pos = pos
                            best_cell = cell
                else:
                    best_pos = random.randrange(len(empties))
                    best_cell = empties[best_pos]

                empties[best_pos] = empties[-1]
                empties.pop()

                sim_board.play_unchecked(best_cell, p)
                if p == 1:
                    if not black_seen[best_cell]:
                        black_seen[best_cell] = 1
                        black_moves.append(best_cell)
                else:
                    if not white_seen[best_cell]:
                        white_seen[best_cell] = 1
                        white_moves.append(best_cell)
                p = 3 - p

            # On a full Y board there is exactly one winner.
            winner = 1 if sim_board.check_win(1) else 2

            # Backpropagation
            current = node
            while current is not None:
                current.visits += 1
                if current.player == winner:
                    current.wins += 1.0

                if use_rave and current.player is not None:
                    next_player = 3 - current.player
                    moves = black_moves if next_player == 1 else white_moves
                    rave_visits = current.rave_visits
                    rave_wins = current.rave_wins
                    is_win = winner == next_player
                    for mv in moves:
                        rave_visits[mv] = rave_visits.get(mv, 0) + 1
                        if is_win:
                            rave_wins[mv] = rave_wins.get(mv, 0.0) + 1.0

                current = current.parent

        if not root.children:
            return random.choice(empty)

        best = max(root.children, key=lambda child: child.visits)
        return best.move


def play_game(size=9, black_agent=None, white_agent=None, verbose=False):
    """Play a complete game of Y between two agents."""
    board = YBoard(size)
    current = Player.BLACK

    while True:
        if current == Player.BLACK:
            move = black_agent.select_move(board, current)
        else:
            move = white_agent.select_move(board, current)

        success = board.play(move, int(current))
        assert success, f"Illegal move {move} by {'BLACK' if current == Player.BLACK else 'WHITE'}"

        if verbose:
            r, c = board.idx_to_rc(move)
            name = "BLACK" if current == Player.BLACK else "WHITE"
            print(f"{name} plays ({r},{c})")
            board.display()

        if board.check_win(int(current)):
            if verbose:
                name = "BLACK" if current == Player.BLACK else "WHITE"
                print(f"{name} wins!")
            return current

        current = current.opponent


class RandomAgent:
    """Random baseline player."""

    def select_move(self, board, player):
        return random.choice(board.get_empty_cells())
