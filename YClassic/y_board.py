"""
Y board implementation on a regular triangular hex grid.

Y is played on a triangular board of hex-connected cells. A player wins by
forming a connected group that touches all three sides of the triangle. Corner
cells count as belonging to both adjacent sides.

This code uses the regular-triangle variant rather than the commercial Kadon
board with three pentagons. That is still a standard way to play Y and matches
the triangular grid assumed by the rest of this repository.
"""

from enum import IntEnum


class Player(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self):
        if self == Player.BLACK:
            return Player.WHITE
        if self == Player.WHITE:
            return Player.BLACK
        return Player.EMPTY


# Regular triangular board with hex connectivity.
Y_NEIGHBORS = [
    (0, -1),   # left
    (0, 1),    # right
    (-1, -1),  # up-left
    (-1, 0),   # up-right
    (1, 0),    # down-left
    (1, 1),    # down-right
]


class YBoard:
    """
    Y board with per-player Union-Find connectivity and side masks.

    Components carry a 3-bit side mask:
      LEFT   = 0b001
      RIGHT  = 0b010
      BOTTOM = 0b100

    A component wins as soon as its mask becomes 0b111.
    """

    LEFT = 1
    RIGHT = 2
    BOTTOM = 4
    ALL_SIDES = LEFT | RIGHT | BOTTOM

    def __init__(self, size=9):
        self.size = size
        self.n = size * (size + 1) // 2
        self.board = [0] * self.n
        self.move_count = 0

        # Per-player Union-Find state. Index 0 is unused to allow direct
        # indexing by player id (1 or 2).
        self.parent = [
            None,
            list(range(self.n)),
            list(range(self.n)),
        ]
        self.rank = [
            None,
            [0] * self.n,
            [0] * self.n,
        ]
        self.component_mask = [
            None,
            [0] * self.n,
            [0] * self.n,
        ]
        self._winner = [False, False, False]

        self.row_starts = [0] * size
        self._row_of_idx = [0] * self.n
        self._col_of_idx = [0] * self.n

        idx = 0
        for r in range(size):
            self.row_starts[r] = idx
            for c in range(r + 1):
                self._row_of_idx[idx] = r
                self._col_of_idx[idx] = c
                idx += 1

        self._cell_side_mask = [0] * self.n
        self._neighbors = [[] for _ in range(self.n)]

        for r in range(size):
            for c in range(r + 1):
                idx = self.rc_to_idx(r, c)

                mask = 0
                if c == 0:
                    mask |= self.LEFT
                if c == r:
                    mask |= self.RIGHT
                if r == size - 1:
                    mask |= self.BOTTOM
                self._cell_side_mask[idx] = mask

                nbrs = self._neighbors[idx]
                for dr, dc in Y_NEIGHBORS:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < size and 0 <= nc <= nr:
                        nbrs.append(self.rc_to_idx(nr, nc))

    def _find(self, x, player):
        parent = self.parent[player]
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(self, a, b, player):
        parent = self.parent[player]
        rank = self.rank[player]
        component_mask = self.component_mask[player]

        ra = self._find(a, player)
        rb = self._find(b, player)
        if ra == rb:
            return ra

        if rank[ra] < rank[rb]:
            ra, rb = rb, ra

        parent[rb] = ra
        component_mask[ra] |= component_mask[rb]
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return ra

    def clone(self):
        """Fast deep copy that reuses precomputed read-only tables."""
        new = YBoard.__new__(YBoard)
        new.size = self.size
        new.n = self.n
        new.board = self.board[:]
        new.move_count = self.move_count
        new.parent = [
            None,
            self.parent[1][:],
            self.parent[2][:],
        ]
        new.rank = [
            None,
            self.rank[1][:],
            self.rank[2][:],
        ]
        new.component_mask = [
            None,
            self.component_mask[1][:],
            self.component_mask[2][:],
        ]
        new._winner = self._winner[:]
        new.row_starts = self.row_starts
        new._row_of_idx = self._row_of_idx
        new._col_of_idx = self._col_of_idx
        new._cell_side_mask = self._cell_side_mask
        new._neighbors = self._neighbors
        return new

    def play(self, idx, player):
        """Place a stone at idx if empty. Returns True on success."""
        if self.board[idx] != 0:
            return False
        self.play_unchecked(idx, player)
        return True

    def play_unchecked(self, idx, player):
        """Place a stone without legality checks."""
        self.board[idx] = player
        self.move_count += 1

        component_mask = self.component_mask[player]
        component_mask[idx] = self._cell_side_mask[idx]

        for nidx in self._neighbors[idx]:
            if self.board[nidx] == player:
                self._union(idx, nidx, player)

        root = self._find(idx, player)
        if component_mask[root] == self.ALL_SIDES:
            self._winner[player] = True

    def check_win(self, player):
        return self._winner[player]

    def get_empty_cells(self):
        return [i for i in range(self.n) if self.board[i] == 0]

    def rc_to_idx(self, r, c):
        return self.row_starts[r] + c

    def idx_to_rc(self, idx):
        return self._row_of_idx[idx], self._col_of_idx[idx]

    def display(self):
        symbols = {0: ".", 1: "B", 2: "W"}
        for r in range(self.size):
            indent = " " * (self.size - r - 1)
            row = []
            for c in range(r + 1):
                idx = self.rc_to_idx(r, c)
                row.append(symbols[self.board[idx]])
            print(indent + " ".join(row))
        print()
