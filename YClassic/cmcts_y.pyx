# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-optimized MCTS for Y with UCT + RAVE.

This is the YClassic fast backend. It mirrors HexClassic's architecture:
  - C struct node pool
  - flat RAVE arrays
  - libc rand()-based sampling
  - memcpy board cloning

The stronger rollout policy is Y-specific and scores candidate moves by how
well they merge friendly components and accumulate the three target sides.
"""

from libc.math cimport log as clog, sqrt
from libc.stdlib cimport free, malloc, rand, srand
from libc.string cimport memcpy, memset

import random as py_random

from y_board import Player
from cy_board cimport CYBoard


def seed_rng(int seed):
    """Seed libc rand() for deterministic Cython playouts."""
    srand(seed)


cdef inline int bitcount3(int x) noexcept nogil:
    return (x & 1) + ((x >> 1) & 1) + ((x >> 2) & 1)


cdef inline int rollout_score(CYBoard board, int cell, int player) noexcept nogil:
    cdef int own_mask = board.cell_side_mask[cell]
    cdef int opp_mask = 0
    cdef int own_roots[6]
    cdef int opp_roots[6]
    cdef int own_count = 0
    cdef int opp_count = 0
    cdef int own_neighbors = 0
    cdef int opp_neighbors = 0
    cdef int opp = 3 - player
    cdef int i, j, nidx, stone, root
    cdef bint seen
    cdef int* own_component_mask
    cdef int* opp_component_mask

    if player == 1:
        own_component_mask = board.component_mask1
        opp_component_mask = board.component_mask2
    else:
        own_component_mask = board.component_mask2
        opp_component_mask = board.component_mask1

    for i in range(board.neighbor_count[cell]):
        nidx = board.neighbors[cell][i]
        stone = board.board[nidx]
        if stone == player:
            own_neighbors += 1
            root = board._find(nidx, player)
            seen = False
            for j in range(own_count):
                if own_roots[j] == root:
                    seen = True
                    break
            if not seen:
                own_roots[own_count] = root
                own_count += 1
                own_mask |= own_component_mask[root]
        elif stone == opp:
            opp_neighbors += 1
            root = board._find(nidx, opp)
            seen = False
            for j in range(opp_count):
                if opp_roots[j] == root:
                    seen = True
                    break
            if not seen:
                opp_roots[opp_count] = root
                opp_count += 1
                opp_mask |= opp_component_mask[root]

    cdef int score = 16 * bitcount3(own_mask)
    score += 6 * own_count
    score += 2 * own_neighbors
    score += 4 * bitcount3(board.cell_side_mask[cell])
    score += 3 * bitcount3(opp_mask)
    score += opp_neighbors

    if own_mask == 7:
        score += 1000
    if opp_mask == 7:
        score += 120
    return score


cdef struct MCTSNodeData:
    int move
    int player
    int parent
    int visits
    double wins
    int* children
    int child_count
    int child_capacity
    int untried_start
    int untried_count
    int* rave_visits
    double* rave_wins


cdef inline void node_add_child(MCTSNodeData* node, int child_idx) noexcept nogil:
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


cdef class CMCTSY:
    """Cython MCTS player for Y."""

    cdef int board_size, n_cells, num_simulations, rollout_sample_size
    cdef double c_uct, rave_bias
    cdef bint use_rave, use_connectivity

    def __init__(
        self,
        int board_size=9,
        double c_uct=0.0,
        double rave_bias=0.00025,
        bint use_rave=True,
        int simulation_type=2,
        int num_simulations=10000,
        int rollout_sample_size=6,
    ):
        self.board_size = board_size
        self.n_cells = board_size * (board_size + 1) // 2
        self.c_uct = c_uct
        self.rave_bias = rave_bias
        self.use_rave = use_rave
        self.use_connectivity = simulation_type == 2
        self.num_simulations = num_simulations
        self.rollout_sample_size = rollout_sample_size

    def select_move(self, board_obj, player_obj):
        cdef CYBoard board
        if isinstance(board_obj, CYBoard):
            board = <CYBoard>board_obj
        else:
            board = _python_board_to_cboard(board_obj)

        cdef int player_int = int(player_obj)
        return self._select_move_impl(board, player_int)

    cdef int _select_move_impl(self, CYBoard board, int player_int):
        cdef int opp_int = 3 - player_int
        cdef int n = self.n_cells
        cdef int num_sims = self.num_simulations
        cdef double c_uct = self.c_uct
        cdef double rave_bias = self.rave_bias
        cdef bint use_rave = self.use_rave
        cdef bint use_connectivity = self.use_connectivity
        cdef int rollout_sample_size = self.rollout_sample_size

        cdef list empty_py = board.get_empty_cells()
        cdef int n_empty = len(empty_py)
        if n_empty == 1:
            return empty_py[0]

        cdef int pool_size = 0
        cdef int pool_capacity = num_sims + 2
        cdef MCTSNodeData* nodes = <MCTSNodeData*>malloc(pool_capacity * sizeof(MCTSNodeData))

        cdef int untried_capacity = pool_capacity * n_empty
        cdef int* untried_pool = <int*>malloc(untried_capacity * sizeof(int))
        cdef int untried_pool_used = 0

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

        cdef int i, j, idx, ne
        if use_rave:
            nodes[0].rave_visits = <int*>malloc(n * sizeof(int))
            nodes[0].rave_wins = <double*>malloc(n * sizeof(double))
            memset(nodes[0].rave_visits, 0, n * sizeof(int))
            memset(nodes[0].rave_wins, 0, n * sizeof(double))

        for i in range(n_empty):
            untried_pool[i] = empty_py[i]
        untried_pool_used = n_empty
        pool_size = 1

        cdef int* empties_buf = <int*>malloc(n * sizeof(int))
        cdef int* black_flags = <int*>malloc(n * sizeof(int))
        cdef int* white_flags = <int*>malloc(n * sizeof(int))
        cdef int* black_list = <int*>malloc(n * sizeof(int))
        cdef int* white_list = <int*>malloc(n * sizeof(int))

        cdef int node_idx, cur, sim_idx, parent_idx
        cdef int child_idx, best_child_idx
        cdef int best_move = -1
        cdef int best_visits = -1
        cdef double best_val, val, mean_value, coef, rave_value, log_pv
        cdef int child_visits, parent_visits, rave_count
        cdef CYBoard sim_board
        cdef int move, ui, winner, p, cell
        cdef int remaining, sample_count
        cdef int best_sample_idx, best_score, score
        cdef int black_count, white_count

        for sim_idx in range(num_sims):
            node_idx = root
            sim_board = board.clone()
            cur = player_int

            memset(black_flags, 0, n * sizeof(int))
            memset(white_flags, 0, n * sizeof(int))
            black_count = 0
            white_count = 0

            # Selection
            while nodes[node_idx].untried_count == 0 and nodes[node_idx].child_count > 0:
                best_val = -1e18
                best_child_idx = -1
                parent_visits = nodes[node_idx].visits

                if use_rave:
                    log_pv = clog(<double>parent_visits) if parent_visits > 0 else 0.0
                    for i in range(nodes[node_idx].child_count):
                        child_idx = nodes[node_idx].children[i]
                        child_visits = nodes[child_idx].visits
                        move = nodes[child_idx].move

                        if child_visits == 0:
                            rave_count = nodes[node_idx].rave_visits[move] if nodes[node_idx].rave_visits != NULL else 0
                            if rave_count > 0:
                                val = nodes[node_idx].rave_wins[move] / <double>rave_count
                            else:
                                val = 1e18
                        else:
                            mean_value = nodes[child_idx].wins / <double>child_visits
                            rave_count = nodes[node_idx].rave_visits[move] if nodes[node_idx].rave_visits != NULL else 0
                            if rave_count > 0:
                                rave_value = nodes[node_idx].rave_wins[move] / <double>rave_count
                                coef = 1.0 - <double>rave_count / (
                                    <double>rave_count +
                                    <double>child_visits +
                                    <double>rave_count * <double>child_visits * rave_bias
                                )
                                if coef < 0.0:
                                    coef = 0.0
                                elif coef > 1.0:
                                    coef = 1.0
                                val = mean_value * coef + (1.0 - coef) * rave_value
                            else:
                                val = mean_value
                            if c_uct > 0.0 and parent_visits > 0:
                                val += c_uct * sqrt(log_pv / <double>child_visits)
                        if val > best_val:
                            best_val = val
                            best_child_idx = child_idx
                else:
                    log_pv = clog(<double>parent_visits) if parent_visits > 0 else 0.0
                    for i in range(nodes[node_idx].child_count):
                        child_idx = nodes[node_idx].children[i]
                        child_visits = nodes[child_idx].visits
                        if child_visits == 0:
                            val = 1e18
                        else:
                            val = nodes[child_idx].wins / <double>child_visits
                            if c_uct > 0.0 and parent_visits > 0:
                                val += c_uct * sqrt(log_pv / <double>child_visits)
                        if val > best_val:
                            best_val = val
                            best_child_idx = child_idx

                node_idx = best_child_idx
                sim_board.play_unchecked(nodes[node_idx].move, cur)
                if cur == 1:
                    if black_flags[nodes[node_idx].move] == 0:
                        black_flags[nodes[node_idx].move] = 1
                        black_list[black_count] = nodes[node_idx].move
                        black_count += 1
                else:
                    if white_flags[nodes[node_idx].move] == 0:
                        white_flags[nodes[node_idx].move] = 1
                        white_list[white_count] = nodes[node_idx].move
                        white_count += 1
                cur = 3 - cur

            # Expansion
            if nodes[node_idx].untried_count > 0:
                parent_idx = node_idx
                ui = rand() % nodes[parent_idx].untried_count
                idx = nodes[parent_idx].untried_start + ui
                move = untried_pool[idx]
                untried_pool[idx] = untried_pool[
                    nodes[parent_idx].untried_start + nodes[parent_idx].untried_count - 1
                ]
                nodes[parent_idx].untried_count -= 1

                sim_board.play_unchecked(move, cur)
                if cur == 1:
                    if black_flags[move] == 0:
                        black_flags[move] = 1
                        black_list[black_count] = move
                        black_count += 1
                else:
                    if white_flags[move] == 0:
                        white_flags[move] = 1
                        white_list[white_count] = move
                        white_count += 1

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

                nodes[child_idx].untried_start = untried_pool_used
                ne = 0
                for i in range(n):
                    if sim_board.board[i] == 0:
                        untried_pool[untried_pool_used + ne] = i
                        ne += 1
                nodes[child_idx].untried_count = ne
                untried_pool_used += ne

                nodes[child_idx].rave_visits = NULL
                nodes[child_idx].rave_wins = NULL
                if use_rave:
                    nodes[child_idx].rave_visits = <int*>malloc(n * sizeof(int))
                    nodes[child_idx].rave_wins = <double*>malloc(n * sizeof(double))
                    memset(nodes[child_idx].rave_visits, 0, n * sizeof(int))
                    memset(nodes[child_idx].rave_wins, 0, n * sizeof(double))

                node_add_child(&nodes[parent_idx], child_idx)
                node_idx = child_idx
                cur = 3 - cur

            # Simulation
            remaining = 0
            for i in range(n):
                if sim_board.board[i] == 0:
                    empties_buf[remaining] = i
                    remaining += 1

            p = cur
            while remaining > 0:
                if use_connectivity:
                    sample_count = rollout_sample_size if remaining > rollout_sample_size else remaining
                    if sample_count < remaining:
                        for i in range(sample_count):
                            j = i + rand() % (remaining - i)
                            cell = empties_buf[i]
                            empties_buf[i] = empties_buf[j]
                            empties_buf[j] = cell

                    best_sample_idx = 0
                    best_score = rollout_score(sim_board, empties_buf[0], p)
                    for i in range(1, sample_count):
                        score = rollout_score(sim_board, empties_buf[i], p)
                        if score > best_score:
                            best_score = score
                            best_sample_idx = i
                    cell = empties_buf[best_sample_idx]
                else:
                    best_sample_idx = rand() % remaining
                    cell = empties_buf[best_sample_idx]

                empties_buf[best_sample_idx] = empties_buf[remaining - 1]
                remaining -= 1

                sim_board.play_unchecked(cell, p)
                if p == 1:
                    if black_flags[cell] == 0:
                        black_flags[cell] = 1
                        black_list[black_count] = cell
                        black_count += 1
                else:
                    if white_flags[cell] == 0:
                        white_flags[cell] = 1
                        white_list[white_count] = cell
                        white_count += 1
                p = 3 - p

            winner = 1 if sim_board.check_win(1) else 2

            # Backpropagation
            idx = node_idx
            while idx >= 0:
                nodes[idx].visits += 1
                if nodes[idx].player == winner:
                    nodes[idx].wins += 1.0

                if use_rave and nodes[idx].rave_visits != NULL:
                    p = 3 - nodes[idx].player
                    if p == 1:
                        for i in range(black_count):
                            move = black_list[i]
                            nodes[idx].rave_visits[move] += 1
                            if winner == 1:
                                nodes[idx].rave_wins[move] += 1.0
                    else:
                        for i in range(white_count):
                            move = white_list[i]
                            nodes[idx].rave_visits[move] += 1
                            if winner == 2:
                                nodes[idx].rave_wins[move] += 1.0

                idx = nodes[idx].parent

        for i in range(nodes[root].child_count):
            child_idx = nodes[root].children[i]
            if nodes[child_idx].visits > best_visits:
                best_visits = nodes[child_idx].visits
                best_move = nodes[child_idx].move

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
        free(black_flags)
        free(white_flags)
        free(black_list)
        free(white_list)

        if best_move < 0:
            return empty_py[0]
        return best_move


def _python_board_to_cboard(py_board):
    """Convert a Python YBoard to CYBoard."""
    cdef int size = py_board.size
    cdef CYBoard cb = CYBoard(size)
    cdef int i
    for i in range(cb.n):
        if py_board.board[i] != 0:
            cb.play(i, py_board.board[i])
    return cb


def play_game(int size=9, black_agent=None, white_agent=None, bint verbose=False):
    """Play a complete game of Y."""
    cdef CYBoard board = CYBoard(size)
    cdef int current = 1
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
            r, c = board.idx_to_rc(move)
            name = "BLACK" if current == 1 else "WHITE"
            print(f"{name} plays ({r},{c})")
            board.display()

        if board.check_win(current):
            return Player.BLACK if current == 1 else Player.WHITE

        current = 3 - current


class RandomAgent:
    """Random baseline player."""

    def select_move(self, board, player):
        empty = board.get_empty_cells()
        return py_random.choice(empty)
