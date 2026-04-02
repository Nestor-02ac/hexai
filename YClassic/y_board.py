"""
Y Game Board implementation (triangular grid) with Union-Find.

- Two players
- Goal: connect ALL THREE sides of the triangle
"""

from enum import IntEnum


class Player(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self):
        return Player(3 - self) if self != Player.EMPTY else Player.EMPTY


class YBoard:
    def __init__(self, size=5):
        self.size = size

        # total number of cells = N(N+1)/2
        self.n = size * (size + 1) // 2
        self.board = [0] * self.n
        self.move_count = 0

        # Virtual nodes for the 3 sides
        self.SIDE_A = self.n
        self.SIDE_B = self.n + 1
        self.SIDE_C = self.n + 2
        # Bitmask values for sides : 001 -> touches LEFT, 011 -> touches LEFT + RIGHT, 111 -> WIN
        self.LEFT = 1
        self.RIGHT = 2
        self.BOTTOM = 4

        total = self.n + 3

        # Separate union-find per player
        self.parent = {
            1: list(range(total)),
            2: list(range(total))
        }
        self.rank = {
            1: [0] * total,
            2: [0] * total
        }

        # Precompute row starts
        self.row_starts = []
        idx = 0
        for r in range(size):
            self.row_starts.append(idx)
            idx += (r + 1)

        # Precompute neighbors
        self._neighbors = [[] for _ in range(self.n)]
        for r in range(size):
            for c in range(r + 1):
                idx = self.rc_to_idx(r, c)

                directions = [
                    (r, c - 1),     # left
                    (r, c + 1),     # right
                    (r - 1, c - 1), # up-left
                    (r - 1, c),     # up-right
                    (r + 1, c),     # down-left
                    (r + 1, c + 1), # down-right
                ]

                for nr, nc in directions:
                    if 0 <= nr < size and 0 <= nc <= nr:
                        self._neighbors[idx].append(self.rc_to_idx(nr, nc))

        self._side_a = {self.rc_to_idx(r, 0) for r in range(size)}         # LEFT
        self._side_b = {self.rc_to_idx(r, r) for r in range(size)}          # RIGHT
        self._side_c = {self.rc_to_idx(size - 1, c) for c in range(size)}   # BOTTOM

        self.side_mask = {1: [0] * (self.n + 3), 2: [0] * (self.n + 3)}

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
        mask = self.side_mask[player]

        ra = self._find(a, player)
        rb = self._find(b, player)
        if ra == rb:
            return

        if rank[ra] < rank[rb]:
            ra, rb = rb, ra

        parent[rb] = ra

        # critical: merge side info
        mask[ra] |= mask[rb]

        if rank[ra] == rank[rb]:
            rank[ra] += 1

    def play(self, idx, player):
        if self.board[idx] != 0:
            return False

        self.board[idx] = player
        self.move_count += 1

        # Connect to same-color neighbors
        for nidx in self._neighbors[idx]:
            if self.board[nidx] == player:
                self._union(idx, nidx, player)

        # Compute which sides this move touches
        mask = 0
        if idx in self._side_a:
            mask |= self.LEFT
        if idx in self._side_b:
            mask |= self.RIGHT
        if idx in self._side_c:
            mask |= self.BOTTOM

        # Find root AFTER neighbor unions
        root = self._find(idx, player)

        # Assign side info
        self.side_mask[player][root] |= mask

        return True

    def check_win(self, player):
        for i in range(self.n):
            if self.board[i] == player:
                root = self._find(i, player)
                if self.side_mask[player][root] == 7:
                    return True
        return False

    def get_empty_cells(self):
        return [i for i in range(self.n) if self.board[i] == 0]

    def rc_to_idx(self, r, c):
        return self.row_starts[r] + c

    def idx_to_rc(self, idx):
        for r in range(self.size):
            start = self.row_starts[r]
            end = start + r + 1
            if start <= idx < end:
                return r, idx - start

    def clone(self):
        new = YBoard.__new__(YBoard)

        new.size = self.size
        new.n = self.n
        new.board = self.board[:]
        new.move_count = self.move_count

        new.parent = {
            1: self.parent[1][:],
            2: self.parent[2][:]
        }
        new.rank = {
            1: self.rank[1][:],
            2: self.rank[2][:]
        }

        # Side tracking (CRUCIAL for Y)
        new.side_mask = {
            1: self.side_mask[1][:],
            2: self.side_mask[2][:]
        }

        new.row_starts = self.row_starts
        new._neighbors = self._neighbors
        new._side_a = self._side_a
        new._side_b = self._side_b
        new._side_c = self._side_c

        # Bit flags 
        new.LEFT = self.LEFT
        new.RIGHT = self.RIGHT
        new.BOTTOM = self.BOTTOM

        return new

    def display(self):
        symbols = {0: '.', 1: 'B', 2: 'W'}
        for r in range(self.size):
            indent = " " * (self.size - r - 1)
            row = []
            for c in range(r + 1):
                idx = self.rc_to_idx(r, c)
                row.append(symbols[self.board[idx]])
            print(indent + " ".join(row))
        print()


if __name__ == "__main__":

    board = YBoard(size=5)

    board.play(board.rc_to_idx(0, 0), Player.BLACK)  # touches L+R
    board.play(board.rc_to_idx(2, 0), Player.BLACK)  # touches L
    board.play(board.rc_to_idx(4, 1), Player.BLACK)  # touches B
    board.play(board.rc_to_idx(4, 4), Player.BLACK)  # touches R+B

    board.play(board.rc_to_idx(1, 1), Player.WHITE)
    board.play(board.rc_to_idx(2, 2), Player.WHITE)
    board.play(board.rc_to_idx(3, 2), Player.WHITE)
    board.play(board.rc_to_idx(3, 3), Player.WHITE)

    board.display()

    print("BLACK wins?", board.check_win(Player.BLACK))  # False
    print("WHITE wins?", board.check_win(Player.WHITE))  # False