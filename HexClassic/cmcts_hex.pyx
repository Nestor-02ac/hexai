# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-optimized MCTS for Hex with UCT + RAVE (type 2 rollouts).

Drop-in replacement for mcts_hex.py — same algorithm and API (select_move,
play_game, RandomAgent), ~12x faster on 11x11 boards. Used as the default
backend for running the comparative experiments (--cython flag).

Optimizations over the Python version:
  - C struct node pool with per-node dynamically allocated children arrays.
  - Flat int*/double* arrays for RAVE statistics (vs Python dicts).
  - Fisher-Yates shuffle using libc rand() instead of random.shuffle().
  - Rollout simulation with C-level bridge detection (no Python overhead).
  - Board cloning via memcpy (chex_board.pyx) instead of deepcopy.

Multiprocessing note: call seed_rng(seed) per worker to seed libc rand().
Without this, forked workers share identical rand() state and produce
correlated games that degrade statistical validity.

Build: python setup.py build_ext --inplace (requires Cython).
"""

from libc.stdlib cimport malloc, free, rand, srand, RAND_MAX
from libc.string cimport memset, memcpy
from libc.math cimport sqrt, log as clog

import random as py_random
from hex_board import Player
from chex_board cimport CHexBoard


def seed_rng(int seed):
    """Seed libc rand() — must be called per-worker for multiprocessing."""
    srand(seed)


# Fisher-Yates shuffle using libc rand

cdef inline void shuffle_array(int* arr, int n) noexcept nogil:
    cdef int i, j, tmp
    for i in range(n - 1, 0, -1):
        j = rand() % (i + 1)
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp


# MCTS node struct
# Children stored per-node in a dynamically allocated array.

cdef struct MCTSNodeData:
    int move            # cell index
    int player          # player who placed that move
    int parent          # index in node pool (-1 = root)
    int visits
    double wins
    # Children: dynamically allocated array of node indices
    int* children       # array of child node indices
    int child_count
    int child_capacity
    # Untried moves
    int untried_start   # index in untried_pool
    int untried_count
    # RAVE: flat arrays indexed by cell
    int* rave_visits    # [n_cells] or NULL
    double* rave_wins   # [n_cells] or NULL


cdef inline void node_add_child(MCTSNodeData* node, int child_idx) noexcept nogil:
    """Add a child to node, growing array if needed."""
    cdef int new_cap
    cdef int* new_arr
    if node.child_count >= node.child_capacity:
        new_cap = node.child_capacity * 2 if node.child_capacity > 0 else 8
        new_arr = <int*>malloc(new_cap * sizeof(int))
        if node.children != NULL:
            memcpy(new_arr, node.children, node.child_count * sizeof(int))
            free(node.children)
        node.children = new_arr
        node.child_capacity = new_cap
    node.children[node.child_count] = child_idx
    node.child_count += 1


cdef class CMCTSHex:
    """
    Cython MCTS player for Hex with UCT + RAVE.
    Uses C-level node pool and typed inner loops.
    """
    cdef int board_size, n_cells, num_simulations
    cdef double c_uct, rave_bias
    cdef bint use_rave, use_bridges

    def __init__(self, int board_size=11, double c_uct=0.0,
                 double rave_bias=0.00025, bint use_rave=True,
                 int simulation_type=2, int num_simulations=10000):
        self.board_size = board_size
        self.n_cells = board_size * board_size
        self.c_uct = c_uct
        self.rave_bias = rave_bias
        self.use_rave = use_rave
        self.use_bridges = (simulation_type == 2)
        self.num_simulations = num_simulations

    def select_move(self, board_obj, player_obj):
        """
        Run MCTS and return best move. Compatible with play_game() interface.
        board_obj: CHexBoard (Cython) or HexBoard (Python)
        player_obj: Player enum or int
        """
        cdef CHexBoard board

        # Accept both CHexBoard and HexBoard
        if isinstance(board_obj, CHexBoard):
            board = <CHexBoard>board_obj
        else:
            board = _python_board_to_cboard(board_obj)

        cdef int player_int = int(player_obj)
        return self._select_move_impl(board, player_int)

    cdef int _select_move_impl(self, CHexBoard board, int player_int):
        cdef int opp_int = 3 - player_int
        cdef int n = self.n_cells
        cdef int num_sims = self.num_simulations
        cdef double c_uct = self.c_uct
        cdef double rave_bias = self.rave_bias
        cdef bint use_rave = self.use_rave
        cdef bint use_bridges = self.use_bridges

        # Get empty cells
        cdef list empty_py = board.get_empty_cells()
        cdef int n_empty = len(empty_py)

        if n_empty == 1:
            return empty_py[0]

        # Node pool
        cdef int pool_size = 0
        cdef int pool_capacity = num_sims + 2
        cdef MCTSNodeData* nodes = <MCTSNodeData*>malloc(pool_capacity * sizeof(MCTSNodeData))

        # Untried moves pool
        cdef int untried_capacity = pool_capacity * n_empty
        cdef int* untried_pool = <int*>malloc(untried_capacity * sizeof(int))
        cdef int untried_pool_used = 0

        # Create root node
        cdef int root = 0
        nodes[0].move = -1
        nodes[0].player = opp_int
        nodes[0].parent = -1
        nodes[0].visits = 0
        nodes[0].wins = 0.0
        nodes[0].children = NULL
        nodes[0].child_count = 0
        nodes[0].child_capacity = 0
        nodes[0].untried_start = 0
        nodes[0].untried_count = n_empty
        nodes[0].rave_visits = NULL
        nodes[0].rave_wins = NULL
        if use_rave:
            nodes[0].rave_visits = <int*>malloc(n * sizeof(int))
            nodes[0].rave_wins = <double*>malloc(n * sizeof(double))
            memset(nodes[0].rave_visits, 0, n * sizeof(int))
            memset(nodes[0].rave_wins, 0, n * sizeof(double))

        # Fill root's untried moves
        cdef int i
        for i in range(n_empty):
            untried_pool[i] = empty_py[i]
        untried_pool_used = n_empty
        pool_size = 1

        # Temp arrays for simulation
        cdef int* empties_buf = <int*>malloc(n * sizeof(int))
        cdef int* remaining_flags = <int*>malloc(n * sizeof(int))
        cdef int* saves_buf = <int*>malloc(20 * sizeof(int))
        cdef int* black_amaf = <int*>malloc(n * sizeof(int))
        cdef int* white_amaf = <int*>malloc(n * sizeof(int))

        cdef int node_idx, cur, sim_idx, parent_idx
        cdef int child_idx, best_child_idx
        cdef double best_val, val, m, coef, rw_val
        cdef int cv, pv, rc
        cdef double log_pv
        cdef CHexBoard sim_board
        cdef int move, ne, idx, j, ui
        cdef int winner
        cdef int p, cell, opp_p, nidx, save_count, save, pidx, s1idx, s2idx
        cdef int v1, v2

        for sim_idx in range(num_sims):
            node_idx = root
            sim_board = board.clone()
            cur = player_int

            # Clear AMAF
            memset(black_amaf, 0, n * sizeof(int))
            memset(white_amaf, 0, n * sizeof(int))

            # Selection
            while nodes[node_idx].untried_count == 0 and nodes[node_idx].child_count > 0:
                best_val = -1e18
                best_child_idx = -1
                pv = nodes[node_idx].visits

                if use_rave:
                    log_pv = clog(<double>pv) if pv > 0 else 0.0
                    for i in range(nodes[node_idx].child_count):
                        child_idx = nodes[node_idx].children[i]
                        cv = nodes[child_idx].visits
                        move = nodes[child_idx].move

                        if cv == 0:
                            rc = nodes[node_idx].rave_visits[move] if nodes[node_idx].rave_visits != NULL else 0
                            if rc > 0:
                                val = nodes[node_idx].rave_wins[move] / <double>rc
                            else:
                                val = 1e18
                        else:
                            m = nodes[child_idx].wins / <double>cv
                            rc = nodes[node_idx].rave_visits[move] if nodes[node_idx].rave_visits != NULL else 0
                            if rc > 0:
                                rw_val = nodes[node_idx].rave_wins[move] / <double>rc
                                coef = 1.0 - <double>rc / (<double>rc + <double>cv + <double>rc * <double>cv * rave_bias)
                                if coef < 0.0:
                                    coef = 0.0
                                elif coef > 1.0:
                                    coef = 1.0
                                val = m * coef + (1.0 - coef) * rw_val
                            else:
                                val = m
                            if c_uct > 0.0 and pv > 0:
                                val = val + c_uct * sqrt(log_pv / <double>cv)
                        if val > best_val:
                            best_val = val
                            best_child_idx = child_idx
                else:
                    log_pv = clog(<double>pv) if pv > 0 else 0.0
                    for i in range(nodes[node_idx].child_count):
                        child_idx = nodes[node_idx].children[i]
                        cv = nodes[child_idx].visits
                        if cv == 0:
                            val = 1e18
                        else:
                            val = nodes[child_idx].wins / <double>cv
                            if c_uct > 0.0:
                                val = val + c_uct * sqrt(log_pv / <double>cv)
                        if val > best_val:
                            best_val = val
                            best_child_idx = child_idx

                node_idx = best_child_idx
                sim_board.play_unchecked(nodes[node_idx].move, cur)
                if cur == 1:
                    black_amaf[nodes[node_idx].move] = 1
                else:
                    white_amaf[nodes[node_idx].move] = 1
                cur = 3 - cur

            # Expansion
            if nodes[node_idx].untried_count > 0:
                parent_idx = node_idx

                # Pick random untried move (swap-with-last removal)
                ui = rand() % nodes[parent_idx].untried_count
                idx = nodes[parent_idx].untried_start + ui
                move = untried_pool[idx]
                untried_pool[idx] = untried_pool[nodes[parent_idx].untried_start + nodes[parent_idx].untried_count - 1]
                nodes[parent_idx].untried_count -= 1

                sim_board.play_unchecked(move, cur)
                if cur == 1:
                    black_amaf[move] = 1
                else:
                    white_amaf[move] = 1

                # Create child node
                if pool_size < pool_capacity:
                    child_idx = pool_size
                    pool_size += 1

                    nodes[child_idx].move = move
                    nodes[child_idx].player = cur
                    nodes[child_idx].parent = parent_idx
                    nodes[child_idx].visits = 0
                    nodes[child_idx].wins = 0.0
                    nodes[child_idx].children = NULL
                    nodes[child_idx].child_count = 0
                    nodes[child_idx].child_capacity = 0

                    # Collect untried moves for child
                    nodes[child_idx].untried_start = untried_pool_used
                    ne = 0
                    for i in range(n):
                        if sim_board.board[i] == 0:
                            untried_pool[untried_pool_used + ne] = i
                            ne += 1
                    nodes[child_idx].untried_count = ne
                    untried_pool_used += ne

                    # RAVE arrays
                    nodes[child_idx].rave_visits = NULL
                    nodes[child_idx].rave_wins = NULL
                    if use_rave:
                        nodes[child_idx].rave_visits = <int*>malloc(n * sizeof(int))
                        nodes[child_idx].rave_wins = <double*>malloc(n * sizeof(double))
                        memset(nodes[child_idx].rave_visits, 0, n * sizeof(int))
                        memset(nodes[child_idx].rave_wins, 0, n * sizeof(double))

                    # Register child under parent
                    node_add_child(&nodes[parent_idx], child_idx)

                    node_idx = child_idx
                cur = 3 - cur

            # Simulation
            ne = 0
            for i in range(n):
                if sim_board.board[i] == 0:
                    empties_buf[ne] = i
                    ne += 1
            shuffle_array(empties_buf, ne)

            if use_bridges:
                memset(remaining_flags, 0, n * sizeof(int))
                for i in range(ne):
                    remaining_flags[empties_buf[i]] = 1

                p = cur
                for i in range(ne):
                    cell = empties_buf[i]
                    if remaining_flags[cell] == 0:
                        continue
                    remaining_flags[cell] = 0

                    sim_board.play_unchecked(cell, p)
                    if p == 1:
                        black_amaf[cell] = 1
                    else:
                        white_amaf[cell] = 1

                    # Bridge defense for opponent
                    opp_p = 3 - p
                    save_count = 0
                    for j in range(sim_board.neighbor_count[cell]):
                        nidx = sim_board.neighbors[cell][j]
                        if sim_board.board[nidx] == opp_p:
                            for idx in range(sim_board.bridge_count[nidx]):
                                pidx = sim_board.bridge_data[nidx][idx * 3]
                                s1idx = sim_board.bridge_data[nidx][idx * 3 + 1]
                                s2idx = sim_board.bridge_data[nidx][idx * 3 + 2]
                                if sim_board.board[pidx] != opp_p:
                                    continue
                                v1 = sim_board.board[s1idx]
                                v2 = sim_board.board[s2idx]
                                if v1 == p and v2 == 0 and remaining_flags[s2idx]:
                                    saves_buf[save_count] = s2idx
                                    save_count += 1
                                    if save_count >= 20:
                                        break
                                elif v2 == p and v1 == 0 and remaining_flags[s1idx]:
                                    saves_buf[save_count] = s1idx
                                    save_count += 1
                                    if save_count >= 20:
                                        break
                            if save_count >= 20:
                                break

                    if save_count > 0:
                        save = saves_buf[rand() % save_count]
                        remaining_flags[save] = 0
                        sim_board.play_unchecked(save, opp_p)
                        if opp_p == 1:
                            black_amaf[save] = 1
                        else:
                            white_amaf[save] = 1
                        continue  # p's turn again
                    p = opp_p
            else:
                p = cur
                for i in range(ne):
                    cell = empties_buf[i]
                    sim_board.play_unchecked(cell, p)
                    if p == 1:
                        black_amaf[cell] = 1
                    else:
                        white_amaf[cell] = 1
                    p = 3 - p

            # Determine winner
            if sim_board.check_win(1):
                winner = 1
            else:
                winner = 2

            # Backpropagation
            idx = node_idx
            while idx >= 0:
                nodes[idx].visits += 1
                if nodes[idx].player == winner:
                    nodes[idx].wins += 1.0

                if use_rave and nodes[idx].rave_visits != NULL:
                    p = 3 - nodes[idx].player
                    if p == 1:
                        for i in range(n):
                            if black_amaf[i]:
                                nodes[idx].rave_visits[i] += 1
                                if winner == 1:
                                    nodes[idx].rave_wins[i] += 1.0
                    else:
                        for i in range(n):
                            if white_amaf[i]:
                                nodes[idx].rave_visits[i] += 1
                                if winner == 2:
                                    nodes[idx].rave_wins[i] += 1.0
                idx = nodes[idx].parent

        # Find most visited root child
        cdef int best_move = -1
        cdef int best_visits = -1
        for i in range(nodes[root].child_count):
            child_idx = nodes[root].children[i]
            if nodes[child_idx].visits > best_visits:
                best_visits = nodes[child_idx].visits
                best_move = nodes[child_idx].move

        # Cleanup
        for i in range(pool_size):
            if nodes[i].rave_visits != NULL:
                free(nodes[i].rave_visits)
            if nodes[i].rave_wins != NULL:
                free(nodes[i].rave_wins)
            if nodes[i].children != NULL:
                free(nodes[i].children)
        free(nodes)
        free(untried_pool)
        free(empties_buf)
        free(remaining_flags)
        free(saves_buf)
        free(black_amaf)
        free(white_amaf)

        if best_move < 0:
            return empty_py[0]
        return best_move


def _python_board_to_cboard(py_board):
    """Convert a Python HexBoard to CHexBoard."""
    cdef int size = py_board.size
    cdef CHexBoard cb = CHexBoard(size)
    cdef int i
    for i in range(cb.n):
        if py_board.board[i] != 0:
            cb.play(i, py_board.board[i])
    return cb


def play_game(int size=11, black_agent=None, white_agent=None, bint verbose=False):
    """Play a complete game of Hex. Compatible with mcts_hex.play_game()."""
    cdef CHexBoard board = CHexBoard(size)
    cdef int current = 1  # BLACK
    cdef int move
    cdef bint success

    while True:
        if current == 1:
            move = black_agent.select_move(board, Player.BLACK)
        else:
            move = white_agent.select_move(board, Player.WHITE)

        success = board.play(move, current)
        assert success, f"Illegal move {move}"

        if verbose:
            r, c = move // size, move % size
            name = 'BLACK' if current == 1 else 'WHITE'
            print(f"{name} plays ({r},{c})")
            board.display()

        if board.check_win(current):
            if verbose:
                name = 'BLACK' if current == 1 else 'WHITE'
                print(f"{name} wins!")
            return Player.BLACK if current == 1 else Player.WHITE

        current = 3 - current


class RandomAgent:
    """Random player for baseline."""
    def select_move(self, board, player):
        empty = board.get_empty_cells()
        return py_random.choice(empty)
