"""
Monte-Carlo Tree Search for Hex with UCT and RAVE (AMAF).

Based on: "Monte-Carlo Hex" by Cazenave & Saffidine.

Key formulas from the paper:
  UCT: mu + C * sqrt(log(parent_visits) / visits)
  RAVE coef: coef = 1.0 - rc / (rc + c + rc * c * bias)
  RAVE val:  val = m * coef + (1.0 - coef) * rw / rc

Implemented features:
  - UCT tree search with configurable exploration constant
  - RAVE/AMAF heuristic with configurable bias
  - Bridge detection during rollout simulations
  - Union-Find win detection

NOT implemented (documented omissions from the paper):
  - Level-2 edge templates (section 1.1): would require pattern-matching
    for the 4-3-2 edge template during simulations.
  - Ziggurat detection (section 1.1 / 2.1): the paper itself notes this
    makes sequential play slower (32% win rate with fixed time). Would
    benefit from parallelization.
  - Virtual connection solver integration (section 2.3): requires
    Anshelevich's algorithm [1] for computing virtual connections and
    mustplay regions. This is a substantial standalone system. The paper
    reports 69.5% win rate with this addition.

Performance notes:
  - Simulations fill the entire board (Hex has no draws on full boards)
  - Bridge detection only checks neighbors of the last move played
  - Uses flat cell indices instead of (r,c) tuples
"""

import math
import random
from hex_board import HexBoard, Player


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = [
        'move', 'player', 'parent', 'children',
        'visits', 'wins',
        'rave_visits', 'rave_wins',
        'untried_moves',
    ]

    def __init__(self, move=None, player=None, parent=None, untried_moves=None):
        self.move = move          # cell index that led to this node
        self.player = player      # player (int) who just moved
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        # RAVE/AMAF: dict[cell_idx] -> count/wins
        self.rave_visits = {}
        self.rave_wins = {}
        self.untried_moves = untried_moves if untried_moves is not None else []


class SimulationType:
    """
    Types of knowledge used during random simulations.

    Only RANDOM and BRIDGES are implemented. The paper also describes
    level-2 templates and Ziggurats but these are not implemented here
    (see module docstring for rationale).
    """
    RANDOM = 1       # completely random playouts
    BRIDGES = 2      # respond to bridge-breaking moves during playouts


class MCTSHex:
    """
    Monte-Carlo Tree Search player for Hex.
    Implements UCT + RAVE as described in Cazenave & Saffidine's paper.
    """

    def __init__(self, board_size=11, c_uct=0.0, rave_bias=0.00025,
                 use_rave=True, simulation_type=SimulationType.BRIDGES,
                 num_simulations=10000):
        self.board_size = board_size
        self.c_uct = c_uct
        self.rave_bias = rave_bias
        self.use_rave = use_rave
        self.simulation_type = simulation_type
        self.num_simulations = num_simulations

    def select_move(self, board, player):
        """Run MCTS and return the best move (cell index) for player."""
        empty = board.get_empty_cells()
        if len(empty) == 1:
            return empty[0]

        player_int = int(player)
        opp_int = 3 - player_int

        root = MCTSNode(
            player=opp_int,
            untried_moves=list(empty)
        )

        use_rave = self.use_rave
        c_uct = self.c_uct
        rave_bias = self.rave_bias
        use_bridges = (self.simulation_type == SimulationType.BRIDGES)
        log = math.log

        for _ in range(self.num_simulations):
            node = root
            sim_board = board.clone()
            cur = player_int

            # Track tree-path moves for AMAF (fix: include selection moves)
            tree_black_moves = set()
            tree_white_moves = set()

            # === SELECTION ===
            while not node.untried_moves and node.children:
                # Inline select_child for speed
                best_val = -1.0
                best_child = None
                parent_visits = node.visits

                if use_rave:
                    log_pv = log(parent_visits) if parent_visits > 0 else 0
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
                                coef = 1.0 - rc / (rc + cv + rc * cv * rave_bias)
                                if coef < 0.0:
                                    coef = 0.0
                                elif coef > 1.0:
                                    coef = 1.0
                                val = m * coef + (1.0 - coef) * (rw / rc)
                            else:
                                val = m
                            if c_uct > 0 and parent_visits > 0:
                                val += c_uct * math.sqrt(log_pv / cv)
                        if val > best_val:
                            best_val = val
                            best_child = child
                else:
                    log_pv = log(parent_visits) if parent_visits > 0 else 0
                    for child in node.children:
                        cv = child.visits
                        if cv == 0:
                            val = float('inf')
                        else:
                            val = child.wins / cv
                            if c_uct > 0:
                                val += c_uct * math.sqrt(log_pv / cv)
                        if val > best_val:
                            best_val = val
                            best_child = child

                node = best_child
                sim_board.play(node.move, cur)
                # Record tree-path move for AMAF
                if cur == 1:
                    tree_black_moves.add(node.move)
                else:
                    tree_white_moves.add(node.move)
                cur = 3 - cur

            # === EXPANSION ===
            if node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                move = node.untried_moves[idx]
                # Fast removal: swap with last
                node.untried_moves[idx] = node.untried_moves[-1]
                node.untried_moves.pop()

                sim_board.play(move, cur)
                # Record expanded move for AMAF
                if cur == 1:
                    tree_black_moves.add(move)
                else:
                    tree_white_moves.add(move)

                child = MCTSNode(
                    move=move,
                    player=cur,
                    parent=node,
                    untried_moves=[i for i in range(sim_board.n)
                                   if sim_board.board[i] == 0]
                )
                node.children.append(child)
                node = child
                cur = 3 - cur

            # === SIMULATION (rollout) ===
            # Get empty cells, shuffle, fill alternately
            empties = [i for i in range(sim_board.n) if sim_board.board[i] == 0]
            random.shuffle(empties)

            # Start with tree-path moves, add rollout moves on top
            black_sim = set(tree_black_moves)
            white_sim = set(tree_white_moves)

            if use_bridges:
                # Play with bridge defense
                p = cur
                remaining = set(empties)

                for cell in empties:
                    if cell not in remaining:
                        continue
                    remaining.discard(cell)

                    sim_board.play(cell, p)
                    if p == 1:
                        black_sim.add(cell)
                    else:
                        white_sim.add(cell)

                    # Check bridge defense for opponent
                    opp_p = 3 - p
                    saves = []
                    # Only check bridges around the cell just played
                    for nidx in sim_board._neighbors[cell]:
                        if sim_board.board[nidx] == opp_p:
                            for pidx, s1idx, s2idx in sim_board._bridge_patterns[nidx]:
                                if sim_board.board[pidx] != opp_p:
                                    continue
                                v1, v2 = sim_board.board[s1idx], sim_board.board[s2idx]
                                if v1 == p and v2 == 0 and s2idx in remaining:
                                    saves.append(s2idx)
                                elif v2 == p and v1 == 0 and s1idx in remaining:
                                    saves.append(s1idx)
                    if saves:
                        save = saves[random.randrange(len(saves))]
                        remaining.discard(save)
                        sim_board.play(save, opp_p)
                        if opp_p == 1:
                            black_sim.add(save)
                        else:
                            white_sim.add(save)
                        # Opponent responded with bridge save, so it's
                        # still p's turn next (opponent already moved).
                        continue
                    p = opp_p
            else:
                # Pure random fill
                p = cur
                for cell in empties:
                    sim_board.play(cell, p)
                    if p == 1:
                        black_sim.add(cell)
                    else:
                        white_sim.add(cell)
                    p = 3 - p

            # Determine winner (board is full or nearly full)
            if sim_board.check_win(1):
                winner = 1
            else:
                winner = 2

            # === BACKPROPAGATION ===
            # AMAF: update RAVE stats using ALL moves from the full
            # simulation path (tree selection + expansion + rollout),
            # as per "All Moves As First" (Bruegmann 1993, Gelly 2007).
            current = node
            while current is not None:
                current.visits += 1
                if current.player == winner:
                    current.wins += 1.0

                if use_rave and current.player is not None:
                    # Update RAVE for moves of the next player to move
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

        # Choose move with most visits (most robust child)
        if not root.children:
            return random.choice(empty)

        best = max(root.children, key=lambda c: c.visits)
        return best.move


def play_game(size=11, black_agent=None, white_agent=None, verbose=False):
    """Play a complete game of Hex between two agents."""
    board = HexBoard(size)
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
            print(f"{'BLACK' if current == Player.BLACK else 'WHITE'} plays ({r},{c})")
            board.display()

        if board.check_win(int(current)):
            if verbose:
                print(f"{'BLACK' if current == Player.BLACK else 'WHITE'} wins!")
            return current

        current = current.opponent


class RandomAgent:
    """Random player for baseline comparison."""
    def select_move(self, board, player):
        empty = board.get_empty_cells()
        return random.choice(empty)
