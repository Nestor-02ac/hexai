import itertools
import random
import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from YClassic.mcts_y import MCTSY
from YClassic.y_board import Player, YBoard


def naive_win(board, player):
    seen = set()
    for idx in range(board.n):
        cell = board.get_cell(idx) if hasattr(board, "get_cell") else board.board[idx]
        if cell != player or idx in seen:
            continue

        stack = [idx]
        seen.add(idx)
        mask = 0

        while stack:
            cur = stack.pop()
            mask |= board._cell_side_mask[cur] if hasattr(board, "_cell_side_mask") else board.cell_side_mask[cur]

            neighbors = board._neighbors[cur] if hasattr(board, "_neighbors") else [
                board.neighbors[cur][i] for i in range(board.neighbor_count[cur])
            ]
            for nidx in neighbors:
                neighbor_cell = board.get_cell(nidx) if hasattr(board, "get_cell") else board.board[nidx]
                if neighbor_cell == player and nidx not in seen:
                    seen.add(nidx)
                    stack.append(nidx)

        if mask == 7:
            return True
    return False


class TestYBoardLogic(unittest.TestCase):
    def test_full_colorings_have_unique_winner_small_boards(self):
        for size in range(2, 5):
            n = size * (size + 1) // 2
            for cells in itertools.product((1, 2), repeat=n):
                board = YBoard(size)
                board.board[:] = list(cells)
                black_wins = naive_win(board, Player.BLACK)
                white_wins = naive_win(board, Player.WHITE)
                self.assertNotEqual(
                    black_wins,
                    white_wins,
                    msg=f"Expected exactly one winner on size {size} board for coloring {cells}",
                )

    def test_union_find_matches_naive_checker_over_random_games(self):
        rng = random.Random(42)
        for _ in range(50):
            board = YBoard(5)
            moves = board.get_empty_cells()
            rng.shuffle(moves)
            current = Player.BLACK
            for move in moves:
                board.play(move, int(current))
                self.assertEqual(board.check_win(Player.BLACK), naive_win(board, Player.BLACK))
                self.assertEqual(board.check_win(Player.WHITE), naive_win(board, Player.WHITE))
                if board.check_win(int(current)):
                    break
                current = current.opponent

    def test_mcts_returns_legal_move(self):
        board = YBoard(4)
        agent = MCTSY(board_size=4, num_simulations=100)
        move = agent.select_move(board, Player.BLACK)
        self.assertIn(move, board.get_empty_cells())


class TestCythonConsistency(unittest.TestCase):
    def test_cython_board_matches_python_board_when_available(self):
        try:
            from cy_board import CYBoard
        except ModuleNotFoundError:
            self.skipTest("Cython backend not built")

        rng = random.Random(7)
        py_board = YBoard(5)
        cy_board = CYBoard(5)
        moves = py_board.get_empty_cells()
        rng.shuffle(moves)
        current = Player.BLACK

        for move in moves:
            py_board.play(move, int(current))
            cy_board.play(move, int(current))
            self.assertEqual(py_board.check_win(Player.BLACK), cy_board.check_win(Player.BLACK))
            self.assertEqual(py_board.check_win(Player.WHITE), cy_board.check_win(Player.WHITE))
            if py_board.check_win(int(current)):
                break
            current = current.opponent


if __name__ == "__main__":
    unittest.main()
