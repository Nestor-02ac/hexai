# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-optimized Y board with Union-Find connectivity and side masks.

This is the fast backend for YClassic. It mirrors y_board.py's API and keeps
the same regular-triangle board model while moving board state into C arrays.
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from y_board import Player


cdef int[6][2] Y_DIRS
Y_DIRS[0][0] = 0;  Y_DIRS[0][1] = -1
Y_DIRS[1][0] = 0;  Y_DIRS[1][1] = 1
Y_DIRS[2][0] = -1; Y_DIRS[2][1] = -1
Y_DIRS[3][0] = -1; Y_DIRS[3][1] = 0
Y_DIRS[4][0] = 1;  Y_DIRS[4][1] = 0
Y_DIRS[5][0] = 1;  Y_DIRS[5][1] = 1


cdef class CYBoard:
    """Cython Y board with C-array internals."""

    def __cinit__(self, int size=9, bint _skip_init=False):
        self.board = NULL
        self.parent1 = NULL
        self.parent2 = NULL
        self.rank1 = NULL
        self.rank2 = NULL
        self.component_mask1 = NULL
        self.component_mask2 = NULL
        self.row_starts = NULL
        self.row_of_idx = NULL
        self.col_of_idx = NULL
        self.cell_side_mask = NULL
        self.neighbor_count = NULL
        self.neighbors = NULL
        self._owns_precomputed = False
        self.winner1 = False
        self.winner2 = False

    def __init__(self, int size=9, bint _skip_init=False):
        if _skip_init:
            return

        cdef int r, c, i, j, idx, nr, nc, cnt

        self.size = size
        self.n = size * (size + 1) // 2
        self.move_count = 0

        self.board = <int*>malloc(self.n * sizeof(int))
        self.parent1 = <int*>malloc(self.n * sizeof(int))
        self.parent2 = <int*>malloc(self.n * sizeof(int))
        self.rank1 = <int*>malloc(self.n * sizeof(int))
        self.rank2 = <int*>malloc(self.n * sizeof(int))
        self.component_mask1 = <int*>malloc(self.n * sizeof(int))
        self.component_mask2 = <int*>malloc(self.n * sizeof(int))

        memset(self.board, 0, self.n * sizeof(int))
        memset(self.rank1, 0, self.n * sizeof(int))
        memset(self.rank2, 0, self.n * sizeof(int))
        memset(self.component_mask1, 0, self.n * sizeof(int))
        memset(self.component_mask2, 0, self.n * sizeof(int))

        for i in range(self.n):
            self.parent1[i] = i
            self.parent2[i] = i

        self._owns_precomputed = True
        self.row_starts = <int*>malloc(size * sizeof(int))
        self.row_of_idx = <int*>malloc(self.n * sizeof(int))
        self.col_of_idx = <int*>malloc(self.n * sizeof(int))
        self.cell_side_mask = <int*>malloc(self.n * sizeof(int))
        self.neighbor_count = <int*>malloc(self.n * sizeof(int))
        self.neighbors = <int**>malloc(self.n * sizeof(int*))

        for i in range(self.n):
            self.neighbors[i] = <int*>malloc(6 * sizeof(int))

        idx = 0
        for r in range(size):
            self.row_starts[r] = idx
            for c in range(r + 1):
                self.row_of_idx[idx] = r
                self.col_of_idx[idx] = c
                idx += 1

        for r in range(size):
            for c in range(r + 1):
                idx = self.row_starts[r] + c
                self.cell_side_mask[idx] = 0
                if c == 0:
                    self.cell_side_mask[idx] |= 1
                if c == r:
                    self.cell_side_mask[idx] |= 2
                if r == size - 1:
                    self.cell_side_mask[idx] |= 4

                cnt = 0
                for j in range(6):
                    nr = r + Y_DIRS[j][0]
                    nc = c + Y_DIRS[j][1]
                    if 0 <= nr < size and 0 <= nc <= nr:
                        self.neighbors[idx][cnt] = self.row_starts[nr] + nc
                        cnt += 1
                self.neighbor_count[idx] = cnt

    def __dealloc__(self):
        cdef int i
        if self.board != NULL:
            free(self.board)
        if self.parent1 != NULL:
            free(self.parent1)
        if self.parent2 != NULL:
            free(self.parent2)
        if self.rank1 != NULL:
            free(self.rank1)
        if self.rank2 != NULL:
            free(self.rank2)
        if self.component_mask1 != NULL:
            free(self.component_mask1)
        if self.component_mask2 != NULL:
            free(self.component_mask2)
        if self._owns_precomputed:
            if self.row_starts != NULL:
                free(self.row_starts)
            if self.row_of_idx != NULL:
                free(self.row_of_idx)
            if self.col_of_idx != NULL:
                free(self.col_of_idx)
            if self.cell_side_mask != NULL:
                free(self.cell_side_mask)
            if self.neighbor_count != NULL:
                free(self.neighbor_count)
            if self.neighbors != NULL:
                for i in range(self.n):
                    if self.neighbors[i] != NULL:
                        free(self.neighbors[i])
                free(self.neighbors)

    cdef inline int _find(self, int x, int player) noexcept nogil:
        cdef int* parent
        if player == 1:
            parent = self.parent1
        else:
            parent = self.parent2
        cdef int root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    cdef inline void _union(self, int a, int b, int player) noexcept nogil:
        cdef int* parent
        cdef int* rank
        cdef int* component_mask
        if player == 1:
            parent = self.parent1
            rank = self.rank1
            component_mask = self.component_mask1
        else:
            parent = self.parent2
            rank = self.rank2
            component_mask = self.component_mask2
        cdef int ra = self._find(a, player)
        cdef int rb = self._find(b, player)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        component_mask[ra] |= component_mask[rb]
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    cpdef bint play(self, int idx, int player):
        if self.board[idx] != 0:
            return False
        self.play_unchecked(idx, player)
        return True

    cpdef void play_unchecked(self, int idx, int player):
        cdef int i, nidx, cnt, root
        cdef int* component_mask
        if player == 1:
            component_mask = self.component_mask1
        else:
            component_mask = self.component_mask2

        self.board[idx] = player
        self.move_count += 1
        component_mask[idx] = self.cell_side_mask[idx]

        cnt = self.neighbor_count[idx]
        for i in range(cnt):
            nidx = self.neighbors[idx][i]
            if self.board[nidx] == player:
                self._union(idx, nidx, player)

        root = self._find(idx, player)
        if component_mask[root] == 7:
            if player == 1:
                self.winner1 = True
            else:
                self.winner2 = True

    cpdef bint check_win(self, int player):
        return self.winner1 if player == 1 else self.winner2

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

    cpdef int rc_to_idx(self, int r, int c):
        return self.row_starts[r] + c

    cpdef tuple idx_to_rc(self, int idx):
        return self.row_of_idx[idx], self.col_of_idx[idx]

    cpdef CYBoard clone(self):
        cdef CYBoard new_board = CYBoard.__new__(CYBoard, self.size, _skip_init=True)

        new_board.size = self.size
        new_board.n = self.n
        new_board.move_count = self.move_count
        new_board.winner1 = self.winner1
        new_board.winner2 = self.winner2

        new_board.board = <int*>malloc(self.n * sizeof(int))
        new_board.parent1 = <int*>malloc(self.n * sizeof(int))
        new_board.parent2 = <int*>malloc(self.n * sizeof(int))
        new_board.rank1 = <int*>malloc(self.n * sizeof(int))
        new_board.rank2 = <int*>malloc(self.n * sizeof(int))
        new_board.component_mask1 = <int*>malloc(self.n * sizeof(int))
        new_board.component_mask2 = <int*>malloc(self.n * sizeof(int))

        memcpy(new_board.board, self.board, self.n * sizeof(int))
        memcpy(new_board.parent1, self.parent1, self.n * sizeof(int))
        memcpy(new_board.parent2, self.parent2, self.n * sizeof(int))
        memcpy(new_board.rank1, self.rank1, self.n * sizeof(int))
        memcpy(new_board.rank2, self.rank2, self.n * sizeof(int))
        memcpy(new_board.component_mask1, self.component_mask1, self.n * sizeof(int))
        memcpy(new_board.component_mask2, self.component_mask2, self.n * sizeof(int))

        new_board._owns_precomputed = False
        new_board.row_starts = self.row_starts
        new_board.row_of_idx = self.row_of_idx
        new_board.col_of_idx = self.col_of_idx
        new_board.cell_side_mask = self.cell_side_mask
        new_board.neighbor_count = self.neighbor_count
        new_board.neighbors = self.neighbors

        return new_board

    def display(self):
        symbols = {0: ".", 1: "B", 2: "W"}
        cdef int r, c, idx
        for r in range(self.size):
            indent = " " * (self.size - r - 1)
            row = []
            for c in range(r + 1):
                idx = self.row_starts[r] + c
                row.append(symbols[self.board[idx]])
            print(indent + " ".join(row))
        print()
