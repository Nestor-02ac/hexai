# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-optimized Hex board with Union-Find win detection.

Drop-in replacement for hex_board.py's HexBoard. Used by cmcts_hex.pyx for
the fast MCTS backend. Same API: play, check_win, get_empty_cells, clone.

Optimizations over the Python version:
  - C int* arrays for board, parent, rank (vs Python lists).
  - Inline _find/_union with path compression (noexcept nogil).
  - clone() via memcpy; shares precomputed read-only neighbor/bridge data.
  - Precomputed neighbors and bridge patterns per cell at construction time.

Build: python setup.py build_ext --inplace (requires Cython).
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

# Re-export Player for compatibility
from hex_board import Player

# Hex neighbor offsets
cdef int[6][2] HEX_DIRS
HEX_DIRS[0][0] = -1; HEX_DIRS[0][1] = 0
HEX_DIRS[1][0] = -1; HEX_DIRS[1][1] = 1
HEX_DIRS[2][0] = 0;  HEX_DIRS[2][1] = -1
HEX_DIRS[3][0] = 0;  HEX_DIRS[3][1] = 1
HEX_DIRS[4][0] = 1;  HEX_DIRS[4][1] = -1
HEX_DIRS[5][0] = 1;  HEX_DIRS[5][1] = 0

# Precomputed bridge patterns
cdef int NUM_BRIDGE_PATTERNS = 0
cdef int BRIDGE_PATS[20][6]

cdef void _init_bridge_patterns():
    global NUM_BRIDGE_PATTERNS
    cdef int i, j, n1r, n1c, n2r, n2c, dr, dc, br, bc
    cdef int idx = 0
    cdef int nlist[6][2]
    cdef set origin_neighbors = set()

    for i in range(6):
        nlist[i][0] = HEX_DIRS[i][0]
        nlist[i][1] = HEX_DIRS[i][1]
        origin_neighbors.add((HEX_DIRS[i][0], HEX_DIRS[i][1]))

    for i in range(6):
        for j in range(i + 1, 6):
            n1r = nlist[i][0]; n1c = nlist[i][1]
            n2r = nlist[j][0]; n2c = nlist[j][1]
            dr = n2r - n1r
            dc = n2c - n1c
            if (dr, dc) in origin_neighbors:
                br = n1r + n2r
                bc = n1c + n2c
                if ((br - n1r, bc - n1c) in origin_neighbors and
                    (br - n2r, bc - n2c) in origin_neighbors):
                    BRIDGE_PATS[idx][0] = br
                    BRIDGE_PATS[idx][1] = bc
                    BRIDGE_PATS[idx][2] = n1r
                    BRIDGE_PATS[idx][3] = n1c
                    BRIDGE_PATS[idx][4] = n2r
                    BRIDGE_PATS[idx][5] = n2c
                    idx += 1
    NUM_BRIDGE_PATTERNS = idx

_init_bridge_patterns()


cdef class CHexBoard:
    """
    Cython Hex board with C-array internals.
    All cdef attributes are declared in chex_board.pxd.
    """

    def __cinit__(self, int size=11, bint _skip_init=False):
        self.board = NULL
        self.parent = NULL
        self.rank = NULL
        self.neighbor_count = NULL
        self.neighbors = NULL
        self.bridge_count = NULL
        self.bridge_data = NULL
        self._owns_precomputed = False

    def __init__(self, int size=11, bint _skip_init=False):
        if _skip_init:
            return
        cdef int i, j, r, c, nr, nc, idx, total, pidx
        cdef int pr, pc, s1r, s1c, s2r, s2c, s1idx, s2idx, cnt

        self.size = size
        self.n = size * size
        self.move_count = 0

        self.TOP = self.n
        self.BOTTOM = self.n + 1
        self.LEFT = self.n + 2
        self.RIGHT = self.n + 3

        total = self.n + 4
        self.board = <int*>malloc(self.n * sizeof(int))
        self.parent = <int*>malloc(total * sizeof(int))
        self.rank = <int*>malloc(total * sizeof(int))
        memset(self.board, 0, self.n * sizeof(int))
        memset(self.rank, 0, total * sizeof(int))
        for i in range(total):
            self.parent[i] = i

        # Precompute neighbors
        self._owns_precomputed = True
        self.neighbor_count = <int*>malloc(self.n * sizeof(int))
        self.neighbors = <int**>malloc(self.n * sizeof(int*))
        for i in range(self.n):
            self.neighbors[i] = <int*>malloc(6 * sizeof(int))

        for r in range(size):
            for c in range(size):
                idx = r * size + c
                cnt = 0
                for j in range(6):
                    nr = r + HEX_DIRS[j][0]
                    nc = c + HEX_DIRS[j][1]
                    if 0 <= nr < size and 0 <= nc < size:
                        self.neighbors[idx][cnt] = nr * size + nc
                        cnt += 1
                self.neighbor_count[idx] = cnt

        # Precompute bridge patterns per cell
        self.bridge_count = <int*>malloc(self.n * sizeof(int))
        self.bridge_data = <int**>malloc(self.n * sizeof(int*))
        memset(self.bridge_count, 0, self.n * sizeof(int))

        for r in range(size):
            for c in range(size):
                idx = r * size + c
                cnt = 0
                for j in range(NUM_BRIDGE_PATTERNS):
                    pr = r + BRIDGE_PATS[j][0]
                    pc = c + BRIDGE_PATS[j][1]
                    s1r = r + BRIDGE_PATS[j][2]
                    s1c = c + BRIDGE_PATS[j][3]
                    s2r = r + BRIDGE_PATS[j][4]
                    s2c = c + BRIDGE_PATS[j][5]
                    if (0 <= pr < size and 0 <= pc < size and
                        0 <= s1r < size and 0 <= s1c < size and
                        0 <= s2r < size and 0 <= s2c < size):
                        cnt += 1
                self.bridge_count[idx] = cnt

        for r in range(size):
            for c in range(size):
                idx = r * size + c
                cnt = self.bridge_count[idx]
                if cnt > 0:
                    self.bridge_data[idx] = <int*>malloc(cnt * 3 * sizeof(int))
                else:
                    self.bridge_data[idx] = NULL
                cnt = 0
                for j in range(NUM_BRIDGE_PATTERNS):
                    pr = r + BRIDGE_PATS[j][0]
                    pc = c + BRIDGE_PATS[j][1]
                    s1r = r + BRIDGE_PATS[j][2]
                    s1c = c + BRIDGE_PATS[j][3]
                    s2r = r + BRIDGE_PATS[j][4]
                    s2c = c + BRIDGE_PATS[j][5]
                    if (0 <= pr < size and 0 <= pc < size and
                        0 <= s1r < size and 0 <= s1c < size and
                        0 <= s2r < size and 0 <= s2c < size):
                        pidx = pr * size + pc
                        s1idx = s1r * size + s1c
                        s2idx = s2r * size + s2c
                        self.bridge_data[idx][cnt * 3] = pidx
                        self.bridge_data[idx][cnt * 3 + 1] = s1idx
                        self.bridge_data[idx][cnt * 3 + 2] = s2idx
                        cnt += 1

    def __dealloc__(self):
        cdef int i
        if self.board != NULL:
            free(self.board)
        if self.parent != NULL:
            free(self.parent)
        if self.rank != NULL:
            free(self.rank)
        if self._owns_precomputed:
            if self.neighbors != NULL:
                for i in range(self.n):
                    if self.neighbors[i] != NULL:
                        free(self.neighbors[i])
                free(self.neighbors)
            if self.neighbor_count != NULL:
                free(self.neighbor_count)
            if self.bridge_data != NULL:
                for i in range(self.n):
                    if self.bridge_data[i] != NULL:
                        free(self.bridge_data[i])
                free(self.bridge_data)
            if self.bridge_count != NULL:
                free(self.bridge_count)

    cdef inline int _find(self, int x) noexcept nogil:
        cdef int root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    cdef inline void _union(self, int a, int b) noexcept nogil:
        cdef int ra = self._find(a)
        cdef int rb = self._find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    cpdef bint play(self, int idx, int player):
        """Place a stone. Returns True if valid."""
        cdef int i, nidx, cnt
        if self.board[idx] != 0:
            return False
        self.board[idx] = player
        self.move_count += 1

        cnt = self.neighbor_count[idx]
        for i in range(cnt):
            nidx = self.neighbors[idx][i]
            if self.board[nidx] == player:
                self._union(idx, nidx)

        if player == 1:
            if idx < self.size:
                self._union(idx, self.TOP)
            if idx >= self.n - self.size:
                self._union(idx, self.BOTTOM)
        else:
            if idx % self.size == 0:
                self._union(idx, self.LEFT)
            if idx % self.size == self.size - 1:
                self._union(idx, self.RIGHT)
        return True

    cpdef void play_unchecked(self, int idx, int player):
        """Place a stone without validity check (for rollouts)."""
        cdef int i, nidx, cnt
        self.board[idx] = player
        self.move_count += 1

        cnt = self.neighbor_count[idx]
        for i in range(cnt):
            nidx = self.neighbors[idx][i]
            if self.board[nidx] == player:
                self._union(idx, nidx)

        if player == 1:
            if idx < self.size:
                self._union(idx, self.TOP)
            if idx >= self.n - self.size:
                self._union(idx, self.BOTTOM)
        else:
            if idx % self.size == 0:
                self._union(idx, self.LEFT)
            if idx % self.size == self.size - 1:
                self._union(idx, self.RIGHT)

    cpdef bint check_win(self, int player):
        if player == 1:
            return self._find(self.TOP) == self._find(self.BOTTOM)
        else:
            return self._find(self.LEFT) == self._find(self.RIGHT)

    cpdef list get_empty_cells(self):
        cdef list out = []
        cdef int i
        for i in range(self.n):
            if self.board[i] == 0:
                out.append(i)
        return out

    cpdef int get_cell(self, int idx):
        return self.board[idx]

    cpdef set_cell(self, int idx, int val):
        self.board[idx] = val

    cpdef CHexBoard clone(self):
        """Fast deep copy. Shares precomputed neighbor/bridge data."""
        cdef CHexBoard new_board = CHexBoard.__new__(CHexBoard, self.size, _skip_init=True)
        cdef int total = self.n + 4
        new_board.size = self.size
        new_board.n = self.n
        new_board.move_count = self.move_count
        new_board.TOP = self.TOP
        new_board.BOTTOM = self.BOTTOM
        new_board.LEFT = self.LEFT
        new_board.RIGHT = self.RIGHT

        new_board.board = <int*>malloc(self.n * sizeof(int))
        new_board.parent = <int*>malloc(total * sizeof(int))
        new_board.rank = <int*>malloc(total * sizeof(int))
        memcpy(new_board.board, self.board, self.n * sizeof(int))
        memcpy(new_board.parent, self.parent, total * sizeof(int))
        memcpy(new_board.rank, self.rank, total * sizeof(int))

        new_board._owns_precomputed = False
        new_board.neighbor_count = self.neighbor_count
        new_board.neighbors = self.neighbors
        new_board.bridge_count = self.bridge_count
        new_board.bridge_data = self.bridge_data

        return new_board

    def display(self):
        symbols = {0: '.', 1: 'B', 2: 'W'}
        header = "   " + " ".join(f"{c:2d}" for c in range(self.size))
        print(header)
        for r in range(self.size):
            indent = " " * r
            row_str = " ".join(f"{symbols[self.board[r * self.size + c]]:>2}"
                               for c in range(self.size))
            print(f"{indent}{r:2d} {row_str}")
        print()
