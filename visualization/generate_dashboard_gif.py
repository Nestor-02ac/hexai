#!/usr/bin/env python3
"""
Generate a 2x2 dashboard GIF comparing HexClassic and HexGumbel.

Top row:
  Self-play games from HexClassic and HexGumbel.

Bottom row:
  Stylized decision panels for Classic MCTS and Gumbel Zero.

The script keeps the two gameplay rows synchronized by move index. If one game
finishes earlier, its last frame is repeated until the other game ends.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required to render the dashboard. Install it in your env with "
        "`pip install matplotlib` or `conda install matplotlib`."
    ) from exc
import numpy as np
import torch

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, RegularPolygon

HEX_CLASSIC_DIR = REPO_ROOT / "HexClassic"
HEX_GUMBEL_DIR = REPO_ROOT / "HexGumbel"

# None = auto: use the newest checkpoint under HexGumbel/checkpoints.
GUMBEL_CHECKPOINT = "HexGumbel/checkpoints/20260328_180412_b7_c192_r20_s16/iter_0075.pt"
OUTPUT_GIF = "visualization/hex_dashboard_7x7.gif"
BOARD_SIZE = None
CLASSIC_SIMULATIONS = 400
GUMBEL_SIMULATIONS = None
TOP_K_ACTIONS = 6
FRAME_DURATION = 2.20
FIGSIZE = (16, 11)
DPI = 150
FORMULA_FONT_SIZE = 26
CLASSIC_SEED = 7
GUMBEL_SEED = 17
MAX_PLIES = None

COLOR_BG = "#000000"
COLOR_PANEL = "#0A0A0A"
COLOR_PANEL_ALT = "#0C0C0C"
COLOR_PANEL_EDGE = "#2A2A2A"
COLOR_PANEL_GLOW = "#737373"
COLOR_TEXT = "#F4F7FB"
COLOR_MUTED = "#94A3B8"
COLOR_EMPTY_CELL = "#141414"
COLOR_EMPTY_EDGE = "#4A4A4A"
COLOR_BLACK_PLAYER = "#56B6F7"
COLOR_WHITE_PLAYER = "#FF6B6B"
COLOR_CLASSIC = "#56B6F7"
COLOR_GUMBEL = "#14B8A6"
COLOR_ACCENT = "#F6C667"
COLOR_CHIP_BG = "#050505"


@dataclass
class ClassicMoveSnapshot:
    move_index: int
    player: int
    action: int
    board_after: np.ndarray
    winner: int | None
    top_actions: list[dict]
    root_visits: int


@dataclass
class GumbelMoveSnapshot:
    move_index: int
    player: int
    action: int
    board_after: np.ndarray
    winner: int | None
    root_value: float
    top_actions: list[dict]
    improved_policy: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Hex AI comparison dashboard GIF.")
    parser.add_argument("--checkpoint", type=str, default=GUMBEL_CHECKPOINT)
    parser.add_argument("--output", type=str, default=OUTPUT_GIF)
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--classic-sims", type=int, default=CLASSIC_SIMULATIONS)
    parser.add_argument("--gumbel-sims", type=int, default=GUMBEL_SIMULATIONS)
    parser.add_argument("--top-k", type=int, default=TOP_K_ACTIONS)
    parser.add_argument("--duration", type=float, default=FRAME_DURATION)
    parser.add_argument("--formula-font-size", type=float, default=FORMULA_FONT_SIZE)
    parser.add_argument("--classic-seed", type=int, default=CLASSIC_SEED)
    parser.add_argument("--gumbel-seed", type=int, default=GUMBEL_SEED)
    parser.add_argument("--max-plies", type=int, default=MAX_PLIES)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto")
    parser.add_argument("--skip-save", action="store_true", help="Build all frames but skip GIF encoding")
    return parser.parse_args()


def configure_matplotlib_text():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "text.usetex": False,
    })
    if shutil.which("latex") is None:
        return False
    try:
        plt.rcParams["text.usetex"] = True
        fig, ax = plt.subplots(figsize=(1.0, 1.0), dpi=80)
        ax.axis("off")
        ax.text(0.1, 0.5, r"$\pi_g(a\mid s)\propto \exp((\log \pi(a)+g_a)/T)$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        plt.rcParams["text.usetex"] = False
        return False


def import_project_modules(project_dir: Path, module_names: list[str]) -> dict[str, object]:
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(project_dir))
    try:
        importlib.invalidate_caches()
        modules = {name: importlib.import_module(name) for name in module_names}
    finally:
        sys.path = saved_path
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
    return modules


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def find_latest_checkpoint() -> Path:
    checkpoints_root = HEX_GUMBEL_DIR / "checkpoints"
    latest_pointer = checkpoints_root / "latest_checkpoint.txt"
    if latest_pointer.exists():
        latest_rel = latest_pointer.read_text(encoding="utf-8").strip()
        if latest_rel:
            latest_path = resolve_path(latest_rel)
            if latest_path.exists():
                return latest_path

    candidates = list(checkpoints_root.glob("*/iter_*.pt"))
    candidates.extend(checkpoints_root.glob("iter_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under {checkpoints_root}. Train HexGumbel first or pass --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint_path(checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        checkpoint_path = resolve_path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    return find_latest_checkpoint()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    metadata = {
        "checkpoint": None,
        "run": None,
    }
    checkpoint_sidecar = checkpoint_path.with_suffix(".json")
    if checkpoint_sidecar.exists():
        metadata["checkpoint"] = _load_json(checkpoint_sidecar)
    run_metadata_path = checkpoint_path.parent / "run.json"
    if run_metadata_path.exists():
        metadata["run"] = _load_json(run_metadata_path)
    return metadata


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def infer_gumbel_architecture(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    action_size = state_dict["policy_fc.weight"].shape[0]
    board_size = math.isqrt(action_size)
    if board_size * board_size != action_size:
        raise ValueError(f"Invalid checkpoint action size: {action_size}")
    num_channels = state_dict["conv_stem.weight"].shape[0]
    block_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("res_blocks.") and key.endswith("conv1.weight")
    }
    num_res_blocks = len(block_ids)
    return checkpoint, state_dict, board_size, num_channels, num_res_blocks


def resolve_checkpoint_settings(args, checkpoint_path: Path, checkpoint: dict, metadata: dict):
    checkpoint_meta = metadata.get("checkpoint") or {}
    config_meta = checkpoint_meta.get("config") or {}
    network_meta = checkpoint_meta.get("network") or {}
    search_meta = checkpoint_meta.get("search") or {}

    _ckpt, state_dict, inferred_board_size, inferred_channels, inferred_res_blocks = infer_gumbel_architecture(checkpoint_path)
    board_size = coalesce(
        args.board_size,
        config_meta.get("board_size"),
        network_meta.get("board_size"),
        checkpoint.get("board_size"),
        inferred_board_size,
    )
    num_channels = coalesce(
        config_meta.get("num_channels"),
        network_meta.get("num_channels"),
        checkpoint.get("num_channels"),
        inferred_channels,
    )
    num_res_blocks = coalesce(
        config_meta.get("num_res_blocks"),
        network_meta.get("num_res_blocks"),
        checkpoint.get("num_res_blocks"),
        inferred_res_blocks,
    )
    gumbel_sims = coalesce(
        args.gumbel_sims,
        search_meta.get("num_simulations"),
        config_meta.get("num_simulations"),
        GUMBEL_SIMULATIONS,
        board_size * board_size,
    )
    gumbel_sample_size = coalesce(
        search_meta.get("gumbel_sample_size"),
        config_meta.get("gumbel_sample_size"),
        gumbel_sims,
    )
    iteration = coalesce(
        checkpoint_meta.get("iteration"),
        checkpoint.get("iteration"),
        "unknown",
    )
    return {
        "state_dict": state_dict,
        "board_size": int(board_size),
        "num_channels": int(num_channels),
        "num_res_blocks": int(num_res_blocks),
        "gumbel_sims": int(gumbel_sims),
        "gumbel_sample_size": int(gumbel_sample_size),
        "iteration": iteration,
        "metadata_found": checkpoint_meta is not None,
    }


def action_label(action: int, board_size: int) -> str:
    row, col = divmod(action, board_size)
    return f"{row},{col}"


def snapshot_board(board) -> np.ndarray:
    return np.asarray(list(board.board), dtype=np.int8).reshape(board.size, board.size)


def player_display_name(player: int) -> str:
    return "Black / blue" if player == 1 else "White / red"


def select_display_actions(actions: list[dict], top_k: int) -> list[dict]:
    if not actions:
        return []
    if top_k <= 0 or len(actions) <= top_k:
        return list(actions)

    selected_idx = next((idx for idx, item in enumerate(actions) if item["selected"]), None)
    if selected_idx is None or selected_idx < top_k:
        return list(actions[:top_k])
    return list(actions[: top_k - 1]) + [actions[selected_idx]]


def normalize_values(values: list[float], low: float, high: float) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        mid = 0.5 * (low + high)
        return [mid for _ in values]
    scale = (high - low) / (vmax - vmin)
    return [low + (value - vmin) * scale for value in values]


def add_panel_card(ax, facecolor: str, *, transform=None, zorder: float = 0.0):
    if transform is None:
        transform = ax.transAxes

    glow = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        transform=transform,
        facecolor="none",
        edgecolor=COLOR_PANEL_GLOW,
        linewidth=3.0,
        alpha=0.14,
        zorder=zorder,
    )
    card = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        transform=transform,
        facecolor=facecolor,
        edgecolor=COLOR_PANEL_EDGE,
        linewidth=1.4,
        zorder=zorder + 0.01,
    )
    ax.add_patch(glow)
    ax.add_patch(card)


def setup_info_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLOR_BG)
    add_panel_card(ax, COLOR_PANEL_ALT)


def draw_panel_header(ax, title: str, subtitle: str, accent: str):
    ax.text(0.05, 0.91, title, color=COLOR_TEXT, fontsize=17, fontweight="bold", ha="left", va="top")
    ax.text(0.05, 0.82, subtitle, color=COLOR_MUTED, fontsize=9.6, ha="left", va="top")


def draw_formula_text(ax, formula: str, x: float, y: float, color: str, ha: str = "right",
                      fontsize: float = FORMULA_FONT_SIZE):
    ax.text(
        x,
        y,
        formula,
        color=color,
        fontsize=fontsize,
        ha=ha,
        va="top",
    )


def add_metric_chip(ax, x: float, y: float, label: str, value: str, accent: str, width: float = 0.25):
    chip = FancyBboxPatch(
        (x, y),
        width,
        0.10,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=COLOR_CHIP_BG,
        edgecolor=accent,
        linewidth=1.1,
    )
    ax.add_patch(chip)
    ax.text(x + 0.02, y + 0.064, label.upper(), color=COLOR_MUTED, fontsize=8, ha="left", va="center")
    ax.text(x + 0.02, y + 0.030, value, color=COLOR_TEXT, fontsize=11, ha="left", va="center", fontweight="bold")


def run_classic_search(board, player: int, agent, node_cls, simulation_type_cls, rng: random.Random):
    empty = board.get_empty_cells()
    if len(empty) == 1:
        action = empty[0]
        info = [{
            "action": action,
            "label": action_label(action, board.size),
            "visits": 1,
            "q": 0.0,
            "uct": 0.0,
            "selected": True,
        }]
        return action, info, 1

    root = node_cls(player=3 - player, untried_moves=list(empty))
    use_rave = agent.use_rave
    c_uct = agent.c_uct
    rave_bias = agent.rave_bias
    use_bridges = agent.simulation_type == simulation_type_cls.BRIDGES

    for _ in range(agent.num_simulations):
        node = root
        sim_board = board.clone()
        current = player
        tree_black_moves = set()
        tree_white_moves = set()

        while not node.untried_moves and node.children:
            best_value = -1.0
            best_child = None
            parent_visits = node.visits

            if use_rave:
                log_parent = math.log(parent_visits) if parent_visits > 0 else 0.0
                rave_visits = node.rave_visits
                rave_wins = node.rave_wins
                for child in node.children:
                    child_visits = child.visits
                    if child_visits == 0:
                        rave_count = rave_visits.get(child.move, 0)
                        value = rave_wins.get(child.move, 0) / rave_count if rave_count > 0 else float("inf")
                    else:
                        mean = child.wins / child_visits
                        rave_count = rave_visits.get(child.move, 0)
                        if rave_count > 0:
                            rave_value = rave_wins.get(child.move, 0) / rave_count
                            coef = 1.0 - rave_count / (rave_count + child_visits + rave_count * child_visits * rave_bias)
                            coef = min(1.0, max(0.0, coef))
                            value = mean * coef + (1.0 - coef) * rave_value
                        else:
                            value = mean
                        if c_uct > 0 and parent_visits > 0:
                            value += c_uct * math.sqrt(log_parent / child_visits)
                    if value > best_value:
                        best_value = value
                        best_child = child
            else:
                log_parent = math.log(parent_visits) if parent_visits > 0 else 0.0
                for child in node.children:
                    child_visits = child.visits
                    if child_visits == 0:
                        value = float("inf")
                    else:
                        value = child.wins / child_visits
                        if c_uct > 0:
                            value += c_uct * math.sqrt(log_parent / child_visits)
                    if value > best_value:
                        best_value = value
                        best_child = child

            node = best_child
            sim_board.play(node.move, current)
            if current == 1:
                tree_black_moves.add(node.move)
            else:
                tree_white_moves.add(node.move)
            current = 3 - current

        if node.untried_moves:
            move_idx = rng.randrange(len(node.untried_moves))
            move = node.untried_moves[move_idx]
            node.untried_moves[move_idx] = node.untried_moves[-1]
            node.untried_moves.pop()

            sim_board.play(move, current)
            if current == 1:
                tree_black_moves.add(move)
            else:
                tree_white_moves.add(move)

            child = node_cls(
                move=move,
                player=current,
                parent=node,
                untried_moves=[idx for idx in range(sim_board.n) if sim_board.board[idx] == 0],
            )
            node.children.append(child)
            node = child
            current = 3 - current

        empties = [idx for idx in range(sim_board.n) if sim_board.board[idx] == 0]
        rng.shuffle(empties)
        black_sim = set(tree_black_moves)
        white_sim = set(tree_white_moves)

        if use_bridges:
            turn = current
            remaining = set(empties)

            for cell in empties:
                if cell not in remaining:
                    continue
                remaining.discard(cell)
                sim_board.play(cell, turn)
                if turn == 1:
                    black_sim.add(cell)
                else:
                    white_sim.add(cell)

                opponent = 3 - turn
                saves = []
                for neighbor_idx in sim_board._neighbors[cell]:
                    if sim_board.board[neighbor_idx] == opponent:
                        for bridge_idx, save_1, save_2 in sim_board._bridge_patterns[neighbor_idx]:
                            if sim_board.board[bridge_idx] != opponent:
                                continue
                            value_1 = sim_board.board[save_1]
                            value_2 = sim_board.board[save_2]
                            if value_1 == turn and value_2 == 0 and save_2 in remaining:
                                saves.append(save_2)
                            elif value_2 == turn and value_1 == 0 and save_1 in remaining:
                                saves.append(save_1)
                if saves:
                    save = rng.choice(saves)
                    remaining.discard(save)
                    sim_board.play(save, opponent)
                    if opponent == 1:
                        black_sim.add(save)
                    else:
                        white_sim.add(save)
                    continue
                turn = opponent
        else:
            turn = current
            for cell in empties:
                sim_board.play(cell, turn)
                if turn == 1:
                    black_sim.add(cell)
                else:
                    white_sim.add(cell)
                turn = 3 - turn

        winner = 1 if sim_board.check_win(1) else 2

        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if current_node.player == winner:
                current_node.wins += 1.0

            if use_rave and current_node.player is not None:
                next_player = 3 - current_node.player
                moves_set = black_sim if next_player == 1 else white_sim
                for move in moves_set:
                    current_node.rave_visits[move] = current_node.rave_visits.get(move, 0) + 1
                    if winner == next_player:
                        current_node.rave_wins[move] = current_node.rave_wins.get(move, 0.0) + 1.0
            current_node = current_node.parent

    best_child = max(root.children, key=lambda child: child.visits)
    log_parent = math.log(root.visits) if root.visits > 0 else 0.0
    top_actions = []
    for child in root.children:
        visits = child.visits
        q_value = child.wins / visits if visits > 0 else 0.0
        if visits == 0:
            uct_value = float("inf")
        elif c_uct > 0 and root.visits > 0:
            uct_value = q_value + c_uct * math.sqrt(log_parent / visits)
        else:
            uct_value = q_value
        top_actions.append({
            "action": child.move,
            "label": action_label(child.move, board.size),
            "visits": visits,
            "q": q_value,
            "uct": uct_value,
            "selected": child.move == best_child.move,
        })

    top_actions.sort(key=lambda item: (item["visits"], item["q"]), reverse=True)
    return best_child.move, top_actions, root.visits


def play_classic_self_play(classic_board_cls, classic_player_cls, classic_agent_cls, node_cls, simulation_type_cls,
                           board_size: int, num_simulations: int, seed: int, max_plies: int | None):
    rng = random.Random(seed)
    board = classic_board_cls(board_size)
    agent = classic_agent_cls(
        board_size=board_size,
        c_uct=0.0,
        rave_bias=0.00025,
        use_rave=True,
        simulation_type=simulation_type_cls.BRIDGES,
        num_simulations=num_simulations,
    )
    current = classic_player_cls.BLACK
    snapshots = []

    while True:
        action, top_actions, root_visits = run_classic_search(board, int(current), agent, node_cls, simulation_type_cls, rng)
        board.play(action, int(current))
        winner = int(current) if board.check_win(int(current)) else None
        snapshots.append(
            ClassicMoveSnapshot(
                move_index=len(snapshots) + 1,
                player=int(current),
                action=action,
                board_after=snapshot_board(board),
                winner=winner,
                top_actions=top_actions,
                root_visits=root_visits,
            )
        )
        if winner is not None or (max_plies is not None and len(snapshots) >= max_plies):
            break
        current = current.opponent

    return snapshots


def run_gumbel_search(board, player: int, gumbel_mcts, add_noise: bool):
    search = gumbel_mcts.new_search(player)
    state, legal_actions = gumbel_mcts.prepare_expand(search.root, board, player)
    policy_batch, value_batch = gumbel_mcts.evaluate_states([state])
    gumbel_mcts.finish_root(
        search,
        policy_batch[0],
        float(value_batch[0]),
        legal_actions,
        add_noise=add_noise,
    )

    while not gumbel_mcts.search_complete(search):
        leaf_request = gumbel_mcts.simulate_until_leaf(search, board)
        if leaf_request is None:
            continue
        leaf_state, leaf_legal_actions = gumbel_mcts.prepare_expand(
            leaf_request.node,
            leaf_request.board,
            leaf_request.to_play,
        )
        leaf_policy_batch, leaf_value_batch = gumbel_mcts.evaluate_states([leaf_state])
        gumbel_mcts.finish_leaf(
            leaf_request,
            leaf_policy_batch[0],
            float(leaf_value_batch[0]),
            leaf_legal_actions,
        )

    action, improved_policy = gumbel_mcts.finalize_search(search)
    rows = []
    for child in search.root.children.values():
        raw_logit = child.policy_logit - child.policy_noise
        rows.append({
            "action": child.action,
            "label": action_label(child.action, board.size),
            "raw_logit": raw_logit,
            "shifted_logit": child.policy_logit,
            "noise": child.policy_noise,
            "count": child.count,
            "prior": child.prior,
            "improved_policy": float(improved_policy[child.action]),
            "selected": child.action == action,
        })
    rows.sort(key=lambda item: item["shifted_logit"], reverse=True)
    return action, rows, float(search.root_value), improved_policy


def play_gumbel_self_play(gumbel_board_cls, gumbel_player_cls, gumbel_mcts,
                          board_size: int, seed: int, max_plies: int | None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    board = gumbel_board_cls(board_size)
    current = gumbel_player_cls.BLACK
    snapshots = []

    while True:
        action, top_actions, root_value, improved_policy = run_gumbel_search(
            board,
            int(current),
            gumbel_mcts,
            add_noise=True,
        )
        board.play(action, int(current))
        winner = int(current) if board.check_win(int(current)) else None
        snapshots.append(
            GumbelMoveSnapshot(
                move_index=len(snapshots) + 1,
                player=int(current),
                action=action,
                board_after=snapshot_board(board),
                winner=winner,
                root_value=root_value,
                top_actions=top_actions,
                improved_policy=improved_policy.copy(),
            )
        )
        if winner is not None or (max_plies is not None and len(snapshots) >= max_plies):
            break
        current = current.opponent

    return snapshots


def board_geometry(board_size: int):
    centers = {}
    dy = math.sqrt(3) / 2.0
    for row in range(board_size):
        for col in range(board_size):
            x = col + 0.5 * row
            y = -dy * row
            centers[(row, col)] = (x, y)
    return centers


def draw_hex_board(ax, board_state: np.ndarray, title: str, last_action: int | None, player: int | None,
                   winner: int | None):
    board_size = board_state.shape[0]
    centers = board_geometry(board_size)
    radius = 0.56
    colors = {
        0: COLOR_EMPTY_CELL,
        1: COLOR_BLACK_PLAYER,
        2: COLOR_WHITE_PLAYER,
    }
    edge_colors = {
        0: COLOR_EMPTY_EDGE,
        1: "#BFE7FF",
        2: "#FFD1D1",
    }

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLOR_BG)
    add_panel_card(ax, COLOR_PANEL, transform=ax.transAxes, zorder=0)

    board_ax = ax.inset_axes([0.06, 0.12, 0.88, 0.74], zorder=1)
    board_ax.set_aspect("equal")
    board_ax.axis("off")
    board_ax.set_facecolor("none")

    for row in range(board_size):
        for col in range(board_size):
            value = int(board_state[row, col])
            x, y = centers[(row, col)]
            patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=radius,
                orientation=np.radians(30),
                facecolor=colors[value],
                edgecolor=edge_colors[value],
                linewidth=1.25,
            )
            board_ax.add_patch(patch)

    if last_action is not None:
        row, col = divmod(last_action, board_size)
        x, y = centers[(row, col)]
        highlight = Circle((x, y), radius * 0.92, fill=False, linewidth=2.6, edgecolor=COLOR_ACCENT)
        halo = Circle((x, y), radius * 1.12, fill=False, linewidth=1.2, edgecolor=COLOR_ACCENT, alpha=0.55)
        board_ax.add_patch(highlight)
        board_ax.add_patch(halo)

    xs = [x for x, _ in centers.values()]
    ys = [y for _, y in centers.values()]
    min_x = min(xs) - 1.55
    max_x = max(xs) + 1.55
    min_y = min(ys) - 1.45
    max_y = max(ys) + 1.55
    board_ax.set_xlim(min_x, max_x)
    board_ax.set_ylim(min_y, max_y)

    ax.text(0.05, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=16,
            fontweight="bold", color=COLOR_TEXT)
    ax.text(0.05, 0.90, "Black goal: top-bottom", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=COLOR_BLACK_PLAYER)
    ax.text(0.52, 0.90, "White goal: left-right", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=COLOR_WHITE_PLAYER)

    if last_action is not None:
        move_text = f"Last move: {action_label(last_action, board_size)}"
    else:
        move_text = "Last move: -"
    ax.text(0.05, 0.06, move_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=COLOR_MUTED)

    if player is not None and winner is None:
        ax.text(
            0.95,
            0.06,
            f"Played by {player_display_name(player)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=COLOR_TEXT,
        )
    if winner is not None:
        winner_name = player_display_name(winner)
        ax.text(
            0.95,
            0.06,
            f"Winner: {winner_name}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLOR_ACCENT,
        )


def draw_classic_panel(ax, snapshot: ClassicMoveSnapshot, top_k: int, formula_font_size: float):
    displayed = select_display_actions(snapshot.top_actions, top_k)
    selected = next(item for item in snapshot.top_actions if item["selected"])
    setup_info_panel(ax)
    draw_panel_header(ax, "Classic MCTS", "Root action scores before the search commits.", COLOR_CLASSIC)
    formula = r"$q(s,a)+c\,p(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)}$"
    draw_formula_text(ax, formula, x=0.95, y=0.90, color=COLOR_TEXT, ha="right", fontsize=formula_font_size)

    root_x, root_y = 0.13, 0.48
    ax.add_patch(Circle((root_x, root_y), 0.055, facecolor=COLOR_CHIP_BG, edgecolor=COLOR_CLASSIC, linewidth=2.0))
    ax.text(root_x, root_y + 0.004, "s", ha="center", va="center", fontsize=14, color=COLOR_TEXT, fontweight="bold")
    ax.text(root_x, root_y - 0.11, f"N = {snapshot.root_visits}", ha="center", va="center",
            fontsize=9.5, color=COLOR_MUTED)

    max_visits = max(item["visits"] for item in displayed) if displayed else 1
    y_positions = np.linspace(0.69, 0.33, len(displayed)) if displayed else []

    for y, item in zip(y_positions, displayed):
        strength = item["visits"] / max_visits if max_visits > 0 else 0.0
        arrow_end_x = 0.34
        label_x = 0.35
        q_x = 0.42
        lane_start = 0.52
        lane_end = 0.77
        stats_x = 0.92
        edge_color = COLOR_ACCENT if item["selected"] else COLOR_CLASSIC
        line_width = 1.4 + 2.8 * strength
        arrow = FancyArrowPatch(
            (root_x + 0.06, root_y),
            (arrow_end_x, y),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=line_width,
            color=edge_color,
            alpha=0.95 if item["selected"] else 0.60,
        )
        ax.add_patch(arrow)
        ax.text(
            label_x,
            y,
            item["label"],
            color=COLOR_TEXT,
            fontsize=10.2,
            ha="left",
            va="center",
            fontweight="bold" if item["selected"] else "normal",
        )
        ax.text(
            q_x,
            y,
            f"Q {item['q']:+.2f}",
            color=COLOR_MUTED if not item["selected"] else COLOR_TEXT,
            fontsize=9.6,
            ha="left",
            va="center",
            fontweight="bold" if item["selected"] else "normal",
        )
        ax.plot([lane_start, lane_end], [y, y], color=COLOR_PANEL_EDGE, linewidth=6.5, alpha=0.65,
                solid_capstyle="round")
        q_ratio = float(np.clip(item["q"], 0.0, 1.0))
        fill_ratio = max(strength, q_ratio, 0.04)
        fill_end = lane_start + (lane_end - lane_start) * fill_ratio
        ax.plot(
            [lane_start, min(fill_end, lane_end)],
            [y, y],
            color=edge_color,
            linewidth=6.5,
            alpha=0.95,
            solid_capstyle="round",
        )
        ucb_text = "inf" if not math.isfinite(item["uct"]) else f"{item['uct']:+.2f}"
        ax.text(stats_x, y, f"N {item['visits']}  UCB {ucb_text}", color=COLOR_MUTED, fontsize=8.6,
                ha="right", va="center")

    best_q = max(snapshot.top_actions, key=lambda item: item["q"])["q"]
    add_metric_chip(ax, 0.05, 0.08, "selected", selected["label"], COLOR_ACCENT, width=0.24)
    add_metric_chip(ax, 0.32, 0.08, "root visits", str(snapshot.root_visits), COLOR_CLASSIC, width=0.22)
    add_metric_chip(ax, 0.58, 0.08, "best Q", f"{best_q:+.2f}", COLOR_CLASSIC, width=0.18)


def draw_gumbel_panel(ax, snapshot: GumbelMoveSnapshot, top_k: int, formula_font_size: float):
    displayed = select_display_actions(snapshot.top_actions, top_k)
    selected = next(item for item in snapshot.top_actions if item["selected"])
    setup_info_panel(ax)
    draw_panel_header(ax, "Gumbel Zero", "Prior logits after the Gumbel perturbation.", COLOR_GUMBEL)
    formula = r"$\pi_g(a\mid s)\propto e^{(\log \pi(a)+g_a) / T}$"
    draw_formula_text(ax, formula, x=0.95, y=0.90, color=COLOR_TEXT, ha="right", fontsize=formula_font_size)

    ax.text(0.22, 0.66, r"$\log \pi(a)$", color=COLOR_MUTED, fontsize=10.4, ha="center", va="center")
    ax.text(0.77, 0.66, r"$\pi_g(a\mid s)$", color=COLOR_MUTED, fontsize=10.4, ha="center", va="center")

    raw_positions = normalize_values([item["raw_logit"] for item in displayed], 0.18, 0.36)
    y_positions = np.linspace(0.64, 0.33, len(displayed)) if displayed else []

    for y, item, raw_x in zip(y_positions, displayed, raw_positions):
        edge_color = COLOR_ACCENT if item["selected"] else COLOR_GUMBEL
        raw_radius = 0.015 + 0.015 * min(abs(item["raw_logit"]) / 4.0, 1.0)
        bar_x0 = 0.70
        bar_x1 = 0.84
        fill_x1 = bar_x0 + (bar_x1 - bar_x0) * float(np.clip(item["improved_policy"], 0.0, 1.0))
        ax.plot([0.12, 0.91], [y, y], color=COLOR_PANEL_EDGE, linewidth=1.0, alpha=0.40, linestyle="--")
        ax.text(0.06, y, item["label"], color=COLOR_TEXT, fontsize=11, ha="left", va="center",
                fontweight="bold" if item["selected"] else "normal")
        ax.add_patch(Circle((raw_x, y), raw_radius, facecolor="#8B97A8", edgecolor=COLOR_TEXT, linewidth=0.8))
        arrow = FancyArrowPatch(
            (raw_x + raw_radius + 0.012, y),
            (bar_x0 - 0.02, y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8 if item["selected"] else 1.3,
            color=edge_color,
            alpha=0.9 if item["selected"] else 0.65,
        )
        ax.add_patch(arrow)
        ax.text(0.49, y + 0.022, fr"$g_a$ {item['noise']:+.2f}", color=COLOR_MUTED, fontsize=8.9, ha="center", va="center")
        ax.plot([bar_x0, bar_x1], [y, y], color=COLOR_PANEL_EDGE, linewidth=7.0, alpha=0.75, solid_capstyle="round")
        ax.plot([bar_x0, max(fill_x1, bar_x0 + 0.01)], [y, y], color=edge_color, linewidth=7.0, alpha=0.96,
                solid_capstyle="round")
        if item["selected"]:
            ax.plot([bar_x0, max(fill_x1, bar_x0 + 0.01)], [y, y], color=COLOR_ACCENT, linewidth=1.8, alpha=0.95)
        ax.text(0.95, y, fr"$p^*$ {item['improved_policy']:.2f}", color=COLOR_TEXT, fontsize=9.3, ha="right", va="center")

    best_policy = max(snapshot.top_actions, key=lambda item: item["improved_policy"])["improved_policy"]
    add_metric_chip(ax, 0.05, 0.08, "selected", selected["label"], COLOR_ACCENT, width=0.24)
    add_metric_chip(ax, 0.32, 0.08, "root value", f"{snapshot.root_value:+.3f}", COLOR_GUMBEL, width=0.22)
    add_metric_chip(ax, 0.58, 0.08, "best p*", f"{best_policy:.2f}", COLOR_GUMBEL, width=0.18)


def render_dashboard_frame(classic_snapshot: ClassicMoveSnapshot, gumbel_snapshot: GumbelMoveSnapshot,
                           top_k: int, figsize: tuple[float, float], dpi: int,
                           formula_font_size: float):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    fig.patch.set_facecolor(COLOR_BG)
    ax_classic_board, ax_gumbel_board = axes[0]
    ax_classic, ax_gumbel = axes[1]

    classic_title = f"HexClassic self-play,  ply {classic_snapshot.move_index}"
    gumbel_title = f"HexGumbel self-play,  ply {gumbel_snapshot.move_index}"
    draw_hex_board(
        ax_classic_board,
        classic_snapshot.board_after,
        classic_title,
        classic_snapshot.action,
        classic_snapshot.player,
        classic_snapshot.winner,
    )
    draw_hex_board(
        ax_gumbel_board,
        gumbel_snapshot.board_after,
        gumbel_title,
        gumbel_snapshot.action,
        gumbel_snapshot.player,
        gumbel_snapshot.winner,
    )

    draw_classic_panel(ax_classic, classic_snapshot, top_k, formula_font_size)
    draw_gumbel_panel(ax_gumbel, gumbel_snapshot, top_k, formula_font_size)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.04, wspace=0.05, hspace=0.06)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def build_frames(classic_snapshots, gumbel_snapshots, top_k: int, figsize, dpi, formula_font_size: float):
    total_frames = max(len(classic_snapshots), len(gumbel_snapshots))
    frames = []
    for frame_idx in range(total_frames):
        classic_snapshot = classic_snapshots[min(frame_idx, len(classic_snapshots) - 1)]
        gumbel_snapshot = gumbel_snapshots[min(frame_idx, len(gumbel_snapshots) - 1)]
        frames.append(render_dashboard_frame(
            classic_snapshot,
            gumbel_snapshot,
            top_k,
            figsize,
            dpi,
            formula_font_size,
        ))
    return frames


def save_gif(output_path: Path, frames: list[np.ndarray], duration: float):
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "imageio is required to write the GIF. Install it in your env with "
            "`pip install imageio` or `conda install imageio`."
        ) from exc
    imageio.mimsave(output_path, frames, duration=duration, loop=0)


def main():
    args = parse_args()
    use_tex = configure_matplotlib_text()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    metadata = load_checkpoint_metadata(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved = resolve_checkpoint_settings(args, checkpoint_path, checkpoint, metadata)

    classic_modules = import_project_modules(HEX_CLASSIC_DIR, ["hex_board", "mcts_hex"])
    gumbel_modules = import_project_modules(HEX_GUMBEL_DIR, ["hex_board", "config", "neural_net", "mcts"])

    classic_board_cls = classic_modules["hex_board"].HexBoard
    classic_player_cls = classic_modules["hex_board"].Player
    classic_agent_cls = classic_modules["mcts_hex"].MCTSHex
    classic_node_cls = classic_modules["mcts_hex"].MCTSNode
    classic_simulation_type_cls = classic_modules["mcts_hex"].SimulationType

    gumbel_board_cls = gumbel_modules["hex_board"].HexBoard
    gumbel_player_cls = gumbel_modules["hex_board"].Player
    gumbel_config_cls = gumbel_modules["config"].GumbelZeroConfig
    gumbel_net_cls = gumbel_modules["neural_net"].HexNet
    gumbel_mcts_cls = gumbel_modules["mcts"].GumbelMCTS

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gumbel_config = gumbel_config_cls(
        board_size=resolved["board_size"],
        num_channels=resolved["num_channels"],
        num_res_blocks=resolved["num_res_blocks"],
        num_simulations=resolved["gumbel_sims"],
        gumbel_sample_size=resolved["gumbel_sample_size"],
        use_gumbel_noise=True,
        select_action_by_count=False,
        select_action_by_softmax_count=True,
        device=str(device),
    )

    network = gumbel_net_cls(
        board_size=gumbel_config.board_size,
        num_channels=gumbel_config.num_channels,
        num_res_blocks=gumbel_config.num_res_blocks,
    ).to(device)
    network.load_state_dict(resolved["state_dict"])
    network.eval()

    gumbel_mcts = gumbel_mcts_cls(gumbel_config, network, device)

    classic_snapshots = play_classic_self_play(
        classic_board_cls=classic_board_cls,
        classic_player_cls=classic_player_cls,
        classic_agent_cls=classic_agent_cls,
        node_cls=classic_node_cls,
        simulation_type_cls=classic_simulation_type_cls,
        board_size=resolved["board_size"],
        num_simulations=args.classic_sims,
        seed=args.classic_seed,
        max_plies=args.max_plies,
    )

    gumbel_snapshots = play_gumbel_self_play(
        gumbel_board_cls=gumbel_board_cls,
        gumbel_player_cls=gumbel_player_cls,
        gumbel_mcts=gumbel_mcts,
        board_size=resolved["board_size"],
        seed=args.gumbel_seed,
        max_plies=args.max_plies,
    )

    frames = build_frames(
        classic_snapshots=classic_snapshots,
        gumbel_snapshots=gumbel_snapshots,
        top_k=args.top_k,
        figsize=FIGSIZE,
        dpi=DPI,
        formula_font_size=args.formula_font_size,
    )

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_save:
        print(f"Built {len(frames)} dashboard frames. GIF encoding skipped.")
    else:
        save_gif(output_path, frames, duration=args.duration)
        print(f"Saved dashboard GIF to {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metadata sidecar: {'yes' if metadata.get('checkpoint') else 'no'}")
    print(f"Text rendering: {'LaTeX' if use_tex else 'mathtext'}")
    print(f"Classic plies: {len(classic_snapshots)}")
    print(f"Gumbel plies: {len(gumbel_snapshots)}")
    print(f"Checkpoint iteration: {resolved['iteration']}")


if __name__ == "__main__":
    main()