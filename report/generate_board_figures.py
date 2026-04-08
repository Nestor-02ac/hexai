#!/usr/bin/env python3
"""
Generate clean board illustrations for the report.

The figures are rendered from actual self-play end positions so the boards look
like real finished games, but the visual style is adjusted for a white-page
report rather than the dark dashboard GIF.
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Circle


REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

sys.path.insert(0, str(REPO_ROOT / "HexClassic"))
sys.path.insert(0, str(REPO_ROOT / "YClassic"))

from hex_board import HexBoard, Player as HexPlayer  # noqa: E402
from mcts_hex import MCTSHex, SimulationType as HexSimulationType  # noqa: E402
from y_board import YBoard, Player as YPlayer  # noqa: E402
from mcts_y import MCTSY, SimulationType as YSimulationType  # noqa: E402


BLUE = "#2F6BFF"
RED = "#D94B4B"
WIN_RING = "#F2B84B"
GRID = "#D3D7DE"
EMPTY = "#FFFFFF"
TEXT = "#263238"
TRIANGLE = "#5C6773"


def hex_center(r: int, c: int) -> tuple[float, float]:
    return c + 0.5 * r, -math.sqrt(3) * 0.5 * r


def y_center(r: int, c: int) -> tuple[float, float]:
    return c - 0.5 * r, -math.sqrt(3) * 0.5 * r


def play_hex_game(size: int = 7, sims: int = 350, seed: int = 17):
    random.seed(seed)
    board = HexBoard(size)
    agents = {
        HexPlayer.BLACK: MCTSHex(
            board_size=size,
            c_uct=0.0,
            rave_bias=0.00025,
            use_rave=True,
            simulation_type=HexSimulationType.BRIDGES,
            num_simulations=sims,
        ),
        HexPlayer.WHITE: MCTSHex(
            board_size=size,
            c_uct=0.0,
            rave_bias=0.00025,
            use_rave=True,
            simulation_type=HexSimulationType.BRIDGES,
            num_simulations=sims,
        ),
    }

    current = HexPlayer.BLACK
    winner = None
    while winner is None:
        move = agents[current].select_move(board, current)
        board.play(move, int(current))
        if board.check_win(int(current)):
            winner = current
        else:
            current = current.opponent
    return board, winner


def play_y_game(size: int = 7, sims: int = 450, seed: int = 29):
    random.seed(seed)
    board = YBoard(size)
    agents = {
        YPlayer.BLACK: MCTSY(
            board_size=size,
            c_uct=0.0,
            rave_bias=0.00025,
            use_rave=True,
            simulation_type=YSimulationType.CONNECTIVITY,
            num_simulations=sims,
        ),
        YPlayer.WHITE: MCTSY(
            board_size=size,
            c_uct=0.0,
            rave_bias=0.00025,
            use_rave=True,
            simulation_type=YSimulationType.CONNECTIVITY,
            num_simulations=sims,
        ),
    }

    current = YPlayer.BLACK
    winner = None
    while winner is None:
        move = agents[current].select_move(board, current)
        board.play(move, int(current))
        if board.check_win(int(current)):
            winner = current
        else:
            current = current.opponent
    return board, winner


def hex_winning_component(board: HexBoard, winner: HexPlayer) -> set[int]:
    if winner == HexPlayer.BLACK:
        target_root = board._find(board.TOP)
    else:
        target_root = board._find(board.LEFT)
    stones = set()
    for idx, cell in enumerate(board.board):
        if cell == int(winner) and board._find(idx) == target_root:
            stones.add(idx)
    return stones


def y_winning_component(board: YBoard, winner: YPlayer) -> set[int]:
    roots = set()
    for idx, cell in enumerate(board.board):
        if cell == int(winner):
            root = board._find(idx, int(winner))
            if board.component_mask[int(winner)][root] == board.ALL_SIDES:
                roots.add(root)
    stones = set()
    for idx, cell in enumerate(board.board):
        if cell == int(winner) and board._find(idx, int(winner)) in roots:
            stones.add(idx)
    return stones


def _setup_ax(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def render_hex(board: HexBoard, winner: HexPlayer, out_stem: Path):
    fig, ax = _setup_ax((6.8, 6.6))
    winning = hex_winning_component(board, winner)

    # Subtle lattice.
    for idx in range(board.n):
        r, c = board.idx_to_rc(idx)
        x0, y0 = hex_center(r, c)
        for nidx in board._neighbors[idx]:
            if nidx <= idx:
                continue
            nr, nc = board.idx_to_rc(nidx)
            x1, y1 = hex_center(nr, nc)
            ax.plot([x0, x1], [y0, y1], color=GRID, lw=1.2, zorder=1)

    # Colored border hints.
    size = board.size
    top_left = hex_center(0, 0)
    top_right = hex_center(0, size - 1)
    bottom_left = hex_center(size - 1, 0)
    bottom_right = hex_center(size - 1, size - 1)
    ax.plot(
        [top_left[0] - 0.42, top_right[0] + 0.42],
        [top_left[1] + 0.58, top_right[1] + 0.58],
        color=BLUE,
        lw=5,
        solid_capstyle="round",
        zorder=0,
    )
    ax.plot(
        [bottom_left[0] - 0.1, bottom_right[0] + 0.1],
        [bottom_left[1] - 0.58, bottom_right[1] - 0.58],
        color=BLUE,
        lw=5,
        solid_capstyle="round",
        zorder=0,
    )
    ax.plot(
        [top_left[0] - 0.7, bottom_left[0] - 0.7],
        [top_left[1] + 0.12, bottom_left[1] - 0.12],
        color=RED,
        lw=5,
        solid_capstyle="round",
        zorder=0,
    )
    ax.plot(
        [top_right[0] + 0.7, bottom_right[0] + 0.7],
        [top_right[1] + 0.12, bottom_right[1] - 0.12],
        color=RED,
        lw=5,
        solid_capstyle="round",
        zorder=0,
    )

    for idx, cell in enumerate(board.board):
        r, c = board.idx_to_rc(idx)
        x, y = hex_center(r, c)
        color = EMPTY
        edge = "#9FA7B3"
        lw = 1.4
        if cell == int(HexPlayer.BLACK):
            color = BLUE
            edge = "#1D2A57"
            lw = 1.6
        elif cell == int(HexPlayer.WHITE):
            color = RED
            edge = "#5C1F1F"
            lw = 1.6

        if idx in winning:
            ring = Circle((x, y), 0.49, facecolor="none", edgecolor=WIN_RING, lw=3.0, zorder=3)
            ax.add_patch(ring)

        stone = Circle((x, y), 0.39, facecolor=color, edgecolor=edge, lw=lw, zorder=4)
        ax.add_patch(stone)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_stem.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_y(board: YBoard, winner: YPlayer, out_stem: Path):
    fig, ax = _setup_ax((6.8, 6.2))
    winning = y_winning_component(board, winner)

    for idx in range(board.n):
        r, c = board.idx_to_rc(idx)
        x0, y0 = y_center(r, c)
        for nidx in board._neighbors[idx]:
            if nidx <= idx:
                continue
            nr, nc = board.idx_to_rc(nidx)
            x1, y1 = y_center(nr, nc)
            ax.plot([x0, x1], [y0, y1], color=GRID, lw=1.2, zorder=1)

    size = board.size
    top = y_center(0, 0)
    left = y_center(size - 1, 0)
    right = y_center(size - 1, size - 1)
    ax.plot(
        [top[0], left[0] - 0.45],
        [top[1] + 0.55, left[1] - 0.48],
        color=TRIANGLE,
        lw=4.5,
        solid_capstyle="round",
        zorder=0,
    )
    ax.plot(
        [top[0], right[0] + 0.45],
        [top[1] + 0.55, right[1] - 0.48],
        color=TRIANGLE,
        lw=4.5,
        solid_capstyle="round",
        zorder=0,
    )
    ax.plot(
        [left[0] - 0.55, right[0] + 0.55],
        [left[1] - 0.62, right[1] - 0.62],
        color=TRIANGLE,
        lw=4.5,
        solid_capstyle="round",
        zorder=0,
    )

    for idx, cell in enumerate(board.board):
        r, c = board.idx_to_rc(idx)
        x, y = y_center(r, c)
        color = EMPTY
        edge = "#9FA7B3"
        lw = 1.4
        if cell == int(YPlayer.BLACK):
            color = BLUE
            edge = "#1D2A57"
            lw = 1.6
        elif cell == int(YPlayer.WHITE):
            color = RED
            edge = "#5C1F1F"
            lw = 1.6

        if idx in winning:
            ring = Circle((x, y), 0.49, facecolor="none", edgecolor=WIN_RING, lw=3.0, zorder=3)
            ax.add_patch(ring)

        stone = Circle((x, y), 0.39, facecolor=color, edgecolor=edge, lw=lw, zorder=4)
        ax.add_patch(stone)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_stem.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hex_board, hex_winner = play_hex_game()
    y_board, y_winner = play_y_game()
    render_hex(hex_board, hex_winner, FIGURES_DIR / "hex_board_example")
    render_y(y_board, y_winner, FIGURES_DIR / "y_board_example")
    print(f"Saved figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
