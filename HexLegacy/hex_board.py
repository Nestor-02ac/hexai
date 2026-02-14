"""
Hex Board implementation with Union-Find win detection.

Hex is played on an NxN diamond-shaped board of hexagons.
- BLACK (player 1) connects TOP to BOTTOM
- WHITE (player 2) connects LEFT to RIGHT

Each cell (r, c) has 6 neighbors in the hex grid:
  (r-1, c), (r-1, c+1),
  (r, c-1),             (r, c+1),
  (r+1, c-1), (r+1, c)
"""

from enum import IntEnum

class Player(IntEnum):
    EMPTY = 0
    BLACK = 1  # connects top-bottom
    WHITE = 2  # connects left-right

    @property
    def opponent(self):
        if self == Player.BLACK:
            return Player.WHITE
        elif self == Player.WHITE:
            return Player.BLACK
        return Player.EMPTY


# Hex neighbor offsets (row_delta, col_delta)
HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


def _compute_bridge_patterns():
    """
    Compute all bridge patterns from hex geometry.
    A bridge: two same-color stones sharing 2 common empty neighbors.
    Returns list of (bridge_dr, bridge_dc, n1_dr, n1_dc, n2_dr, n2_dc).
    """
    origin_neighbors = set(HEX_NEIGHBORS)
    patterns = []
    neighbor_list = list(HEX_NEIGHBORS)
    for i in range(len(neighbor_list)):
        for j in range(i + 1, len(neighbor_list)):
            n1 = neighbor_list[i]
            n2 = neighbor_list[j]
            dr = n2[0] - n1[0]
            dc = n2[1] - n1[1]
            if (dr, dc) in origin_neighbors:
                br, bc = n1[0] + n2[0], n1[1] + n2[1]
                if ((br - n1[0], bc - n1[1]) in origin_neighbors and
                    (br - n2[0], bc - n2[1]) in origin_neighbors):
                    patterns.append((br, bc, n1[0], n1[1], n2[0], n2[1]))
    return patterns

BRIDGE_PATTERNS = _compute_bridge_patterns()


class HexBoard:
    """
    Hex board using flat arrays and Union-Find for fast win detection.
    Cells are indexed as r * size + c.
    """

    def __init__(self, size=11):
        self.size = size
        self.n = size * size
        # Flat board: 0=empty, 1=black, 2=white
        self.board = [0] * self.n
        self.move_count = 0

        # Virtual nodes for borders
        self.TOP = self.n        # BLACK top
        self.BOTTOM = self.n + 1 # BLACK bottom
        self.LEFT = self.n + 2   # WHITE left
        self.RIGHT = self.n + 3  # WHITE right

        total = self.n + 4
        self.parent = list(range(total))
        self.rank = [0] * total

        # Precompute neighbors for each cell
        self._neighbors = [[] for _ in range(self.n)]
        for r in range(size):
            for c in range(size):
                idx = r * size + c
                for dr, dc in HEX_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        self._neighbors[idx].append(nr * size + nc)

        # Precompute border cells
        self._top_cells = [c for c in range(size)]                     # row 0
        self._bottom_cells = [(size-1)*size + c for c in range(size)]  # row size-1
        self._left_cells = [r * size for r in range(size)]             # col 0
        self._right_cells = [r * size + (size-1) for r in range(size)] # col size-1

        # Precompute bridge patterns for each cell
        self._bridge_patterns = [[] for _ in range(self.n)]
        for r in range(size):
            for c in range(size):
                idx = r * size + c
                for br, bc, n1r, n1c, n2r, n2c in BRIDGE_PATTERNS:
                    pr, pc = r + br, c + bc
                    s1r, s1c = r + n1r, c + n1c
                    s2r, s2c = r + n2r, c + n2c
                    if (0 <= pr < size and 0 <= pc < size and
                        0 <= s1r < size and 0 <= s1c < size and
                        0 <= s2r < size and 0 <= s2c < size):
                        pidx = pr * size + pc
                        s1idx = s1r * size + s1c
                        s2idx = s2r * size + s2c
                        self._bridge_patterns[idx].append((pidx, s1idx, s2idx))

    def _find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def clone(self):
        """Fast deep copy."""
        new = HexBoard.__new__(HexBoard)
        new.size = self.size
        new.n = self.n
        new.board = self.board[:]
        new.move_count = self.move_count
        new.TOP = self.TOP
        new.BOTTOM = self.BOTTOM
        new.LEFT = self.LEFT
        new.RIGHT = self.RIGHT
        new.parent = self.parent[:]
        new.rank = self.rank[:]
        new._neighbors = self._neighbors  # shared (read-only)
        new._top_cells = self._top_cells
        new._bottom_cells = self._bottom_cells
        new._left_cells = self._left_cells
        new._right_cells = self._right_cells
        new._bridge_patterns = self._bridge_patterns  # shared
        return new

    def play(self, idx, player):
        """
        Place a stone. idx = r * size + c, player = 1 (BLACK) or 2 (WHITE).
        Returns True if valid.
        """
        if self.board[idx] != 0:
            return False
        self.board[idx] = player
        self.move_count += 1

        # Connect to same-color neighbors
        for nidx in self._neighbors[idx]:
            if self.board[nidx] == player:
                self._union(idx, nidx)

        # Connect to virtual border nodes
        if player == 1:  # BLACK
            if idx < self.size:  # top row
                self._union(idx, self.TOP)
            if idx >= self.n - self.size:  # bottom row
                self._union(idx, self.BOTTOM)
        else:  # WHITE
            if idx % self.size == 0:  # left column
                self._union(idx, self.LEFT)
            if idx % self.size == self.size - 1:  # right column
                self._union(idx, self.RIGHT)

        return True

    def play_rc(self, r, c, player):
        """Place a stone using (row, col) coordinates."""
        return self.play(r * self.size + c, player)

    def check_win(self, player):
        """Check if player has connected their two borders."""
        if player == 1:  # BLACK
            return self._find(self.TOP) == self._find(self.BOTTOM)
        else:  # WHITE
            return self._find(self.LEFT) == self._find(self.RIGHT)

    def get_empty_cells(self):
        """Return list of indices for empty cells."""
        return [i for i in range(self.n) if self.board[i] == 0]

    def get_bridge_saves(self, idx, player):
        """
        For a stone at idx of given player, find bridge-saving moves.
        Returns list of cell indices that save threatened bridges.
        """
        saves = []
        opp = 3 - player  # opponent: 1->2, 2->1
        for pidx, s1idx, s2idx in self._bridge_patterns[idx]:
            if self.board[pidx] != player:
                continue
            v1, v2 = self.board[s1idx], self.board[s2idx]
            if v1 == opp and v2 == 0:
                saves.append(s2idx)
            elif v2 == opp and v1 == 0:
                saves.append(s1idx)
        return saves

    def idx_to_rc(self, idx):
        return idx // self.size, idx % self.size

    def rc_to_idx(self, r, c):
        return r * self.size + c

    def display(self):
        """Print the board in a readable hex format."""
        symbols = {0: '.', 1: 'B', 2: 'W'}
        header = "   " + " ".join(f"{c:2d}" for c in range(self.size))
        print(header)
        for r in range(self.size):
            indent = " " * r
            row_str = " ".join(f"{symbols[self.board[r * self.size + c]]:>2}"
                               for c in range(self.size))
            print(f"{indent}{r:2d} {row_str}")
        print()
