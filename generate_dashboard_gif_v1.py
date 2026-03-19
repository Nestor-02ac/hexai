#!/usr/bin/env python3
"""
Legacy/original dashboard GIF layout for comparison.

Top row:
  Classic / Gumbel decision plots.

Bottom row:
  Classic / Gumbel gameplay boards.
"""

from __future__ import annotations

import argparse

import generate_dashboard_gif as base
import numpy as np
from matplotlib import pyplot as plt


OUTPUT_GIF_V1 = "hex_dashboard_5x5_v1.gif"


def apply_light_theme():
    base.COLOR_BG = "#FBF8F2"
    base.COLOR_PANEL = "#FFFFFF"
    base.COLOR_PANEL_ALT = "#FCFAF6"
    base.COLOR_PANEL_EDGE = "#D7CDC0"
    base.COLOR_TEXT = "#1F1B18"
    base.COLOR_MUTED = "#6F675D"
    base.COLOR_EMPTY_CELL = "#F3EEE5"
    base.COLOR_EMPTY_EDGE = "#A79D90"
    base.COLOR_CHIP_BG = "#F7F1E7"
    base.COLOR_CLASSIC = "#5B8FD1"
    base.COLOR_GUMBEL = "#1BAA9C"
    base.COLOR_ACCENT = "#D89A2B"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the legacy Hex AI comparison dashboard GIF.")
    parser.add_argument("--checkpoint", type=str, default=base.GUMBEL_CHECKPOINT)
    parser.add_argument("--output", type=str, default=OUTPUT_GIF_V1)
    parser.add_argument("--board-size", type=int, default=base.BOARD_SIZE)
    parser.add_argument("--classic-sims", type=int, default=base.CLASSIC_SIMULATIONS)
    parser.add_argument("--gumbel-sims", type=int, default=base.GUMBEL_SIMULATIONS)
    parser.add_argument("--top-k", type=int, default=base.TOP_K_ACTIONS)
    parser.add_argument("--duration", type=float, default=base.FRAME_DURATION)
    parser.add_argument("--formula-font-size", type=float, default=base.FORMULA_FONT_SIZE)
    parser.add_argument("--classic-seed", type=int, default=base.CLASSIC_SEED)
    parser.add_argument("--gumbel-seed", type=int, default=base.GUMBEL_SEED)
    parser.add_argument("--max-plies", type=int, default=base.MAX_PLIES)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto")
    parser.add_argument("--skip-save", action="store_true", help="Build all frames but skip GIF encoding")
    return parser.parse_args()


def setup_plot_card(ax, title: str):
    ax.set_facecolor(base.COLOR_PANEL_ALT)
    for spine in ax.spines.values():
        spine.set_color(base.COLOR_PANEL_EDGE)
        spine.set_linewidth(1.2)
    ax.tick_params(colors=base.COLOR_MUTED)
    ax.yaxis.label.set_color(base.COLOR_MUTED)
    ax.xaxis.label.set_color(base.COLOR_MUTED)
    ax.title.set_color(base.COLOR_TEXT)
    ax.set_title(title, fontsize=14, fontweight="bold", color=base.COLOR_TEXT, loc="left")


def draw_hex_board_v1(ax, board_state: np.ndarray, title: str, last_action: int | None, player: int | None,
                      winner: int | None):
    board_size = board_state.shape[0]
    centers = base.board_geometry(board_size)
    radius = 0.56
    colors = {
        0: "#F3EEE5",
        1: "#1E1E1E",
        2: "#FAFAFA",
    }
    edge_colors = {
        0: "#B5AB9C",
        1: "#3A3A3A",
        2: "#D7D7D7",
    }

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(base.COLOR_BG)
    card = base.FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        transform=ax.transAxes,
        facecolor=base.COLOR_PANEL,
        edgecolor=base.COLOR_PANEL_EDGE,
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(card)

    for row in range(board_size):
        for col in range(board_size):
            value = int(board_state[row, col])
            x, y = centers[(row, col)]
            patch = base.RegularPolygon(
                (x, y),
                numVertices=6,
                radius=radius,
                orientation=np.radians(30),
                facecolor=colors[value],
                edgecolor=edge_colors[value],
                linewidth=1.0,
            )
            ax.add_patch(patch)

    if last_action is not None:
        row, col = divmod(last_action, board_size)
        x, y = centers[(row, col)]
        highlight = base.Circle((x, y), radius * 0.92, fill=False, linewidth=2.0, edgecolor=base.COLOR_ACCENT)
        halo = base.Circle((x, y), radius * 1.12, fill=False, linewidth=1.0, edgecolor=base.COLOR_ACCENT, alpha=0.5)
        ax.add_patch(highlight)
        ax.add_patch(halo)

    xs = [x for x, _ in centers.values()]
    ys = [y for _, y in centers.values()]
    ax.set_xlim(min(xs) - 1.55, max(xs) + 1.55)
    ax.set_ylim(min(ys) - 1.45, max(ys) + 1.55)

    ax.text(0.05, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=16,
            fontweight="bold", color=base.COLOR_TEXT)
    ax.text(0.05, 0.90, "Black goal: top-bottom", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color="#2B2B2B")
    ax.text(0.52, 0.90, "White goal: left-right", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color="#D8D8D8")

    if last_action is not None:
        move_text = f"Last move: {base.action_label(last_action, board_size)}"
    else:
        move_text = "Last move: -"
    ax.text(0.05, 0.06, move_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=base.COLOR_MUTED)

    if player is not None and winner is None:
        player_name = "Black" if player == 1 else "White"
        ax.text(
            0.95,
            0.06,
            f"Played by {player_name}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=base.COLOR_TEXT,
        )
    if winner is not None:
        winner_name = "Black" if winner == 1 else "White"
        ax.text(
            0.95,
            0.06,
            f"Winner: {winner_name}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=base.COLOR_ACCENT,
        )


def draw_classic_panel_v1(ax, snapshot: base.ClassicMoveSnapshot, top_k: int, formula_font_size: float):
    top_actions = base.select_display_actions(snapshot.top_actions, top_k)
    x = np.arange(len(top_actions))
    visits = [item["visits"] for item in top_actions]
    colors = [base.COLOR_ACCENT if item["selected"] else base.COLOR_CLASSIC for item in top_actions]

    setup_plot_card(ax, "Classic MCTS")
    ax.bar(x, visits, color=colors, alpha=0.95, width=0.66)
    ax.set_xticks(x, [item["label"] for item in top_actions], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Visits", fontsize=10)
    ax.grid(axis="y", alpha=0.18, color=base.COLOR_PANEL_EDGE)

    formula = r"$q(s,a)+c\,p(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)}$"
    ax.text(
        0.02,
        0.98,
        formula,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=formula_font_size,
        color=base.COLOR_TEXT,
    )

    selected = next(item for item in snapshot.top_actions if item["selected"])
    ax.text(
        0.02,
        0.77,
        f"Move {snapshot.move_index} | selected {selected['label']}\n"
        f"root visits: {snapshot.root_visits}\n"
        f"shown actions: {len(top_actions)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=base.COLOR_MUTED,
    )


def draw_gumbel_panel_v1(ax, snapshot: base.GumbelMoveSnapshot, top_k: int, formula_font_size: float):
    top_actions = base.select_display_actions(snapshot.top_actions, top_k)
    x = np.arange(len(top_actions))
    width = 0.36
    raw_logits = [item["raw_logit"] for item in top_actions]
    shifted_logits = [item["shifted_logit"] for item in top_actions]

    setup_plot_card(ax, "Gumbel Zero")
    ax.bar(x - width / 2, raw_logits, width=width, color="#8B97A8", label=r"$\log \pi(a)$", alpha=0.95)
    ax.bar(x + width / 2, shifted_logits, width=width, color=base.COLOR_GUMBEL, label=r"$\log \pi(a)+g_a$", alpha=0.95)

    for idx, item in enumerate(top_actions):
        if item["selected"]:
            ax.bar(x[idx] + width / 2, shifted_logits[idx], width=width, fill=False,
                   edgecolor=base.COLOR_ACCENT, linewidth=2.0)

    ax.set_xticks(x, [item["label"] for item in top_actions], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Logit", fontsize=10)
    ax.grid(axis="y", alpha=0.18, color=base.COLOR_PANEL_EDGE)
    ax.legend(loc="upper right", fontsize=8.5, facecolor=base.COLOR_CHIP_BG, edgecolor=base.COLOR_PANEL_EDGE,
              labelcolor=base.COLOR_TEXT)

    formula = r"$\pi_g(a\mid s)\propto e^{\frac{\log \pi(a)+g_a}{T}}$"
    ax.text(
        0.02,
        0.98,
        formula,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=formula_font_size,
        color=base.COLOR_TEXT,
    )

    selected = next(item for item in snapshot.top_actions if item["selected"])
    ax.text(
        0.02,
        0.77,
        f"Move {snapshot.move_index} | selected {selected['label']}\n"
        f"root value: {snapshot.root_value:+.3f}\n"
        f"shown actions: {len(top_actions)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=base.COLOR_MUTED,
    )


def render_dashboard_frame_v1(
    classic_snapshot: base.ClassicMoveSnapshot,
    gumbel_snapshot: base.GumbelMoveSnapshot,
    top_k: int,
    figsize,
    dpi,
    formula_font_size: float,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    fig.patch.set_facecolor(base.COLOR_BG)

    ax_classic_plot, ax_gumbel_plot = axes[0]
    ax_classic_board, ax_gumbel_board = axes[1]

    draw_classic_panel_v1(ax_classic_plot, classic_snapshot, top_k, formula_font_size)
    draw_gumbel_panel_v1(ax_gumbel_plot, gumbel_snapshot, top_k, formula_font_size)

    draw_hex_board_v1(
        ax_classic_board,
        classic_snapshot.board_after,
        f"HexClassic self-play  |  ply {classic_snapshot.move_index}",
        classic_snapshot.action,
        classic_snapshot.player,
        classic_snapshot.winner,
    )
    draw_hex_board_v1(
        ax_gumbel_board,
        gumbel_snapshot.board_after,
        f"HexGumbel self-play  |  ply {gumbel_snapshot.move_index}",
        gumbel_snapshot.action,
        gumbel_snapshot.player,
        gumbel_snapshot.winner,
    )

    fig.subplots_adjust(left=0.04, right=0.96, top=0.97, bottom=0.04, wspace=0.08, hspace=0.08)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def build_frames_v1(classic_snapshots, gumbel_snapshots, top_k, figsize, dpi, formula_font_size):
    total_frames = max(len(classic_snapshots), len(gumbel_snapshots))
    frames = []
    for frame_idx in range(total_frames):
        classic_snapshot = classic_snapshots[min(frame_idx, len(classic_snapshots) - 1)]
        gumbel_snapshot = gumbel_snapshots[min(frame_idx, len(gumbel_snapshots) - 1)]
        frames.append(
            render_dashboard_frame_v1(
                classic_snapshot,
                gumbel_snapshot,
                top_k,
                figsize,
                dpi,
                formula_font_size,
            )
        )
    return frames


def main():
    apply_light_theme()
    args = parse_args()
    use_tex = base.configure_matplotlib_text()
    checkpoint_path = base.resolve_checkpoint_path(args.checkpoint)
    metadata = base.load_checkpoint_metadata(checkpoint_path)
    checkpoint = base.torch.load(checkpoint_path, map_location="cpu")
    resolved = base.resolve_checkpoint_settings(args, checkpoint_path, checkpoint, metadata)

    classic_modules = base.import_project_modules(base.HEX_CLASSIC_DIR, ["hex_board", "mcts_hex"])
    gumbel_modules = base.import_project_modules(base.HEX_GUMBEL_DIR, ["hex_board", "config", "neural_net", "mcts"])

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
        device = base.torch.device("cuda" if base.torch.cuda.is_available() else "cpu")
    else:
        device = base.torch.device(args.device)

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

    classic_snapshots = base.play_classic_self_play(
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

    gumbel_snapshots = base.play_gumbel_self_play(
        gumbel_board_cls=gumbel_board_cls,
        gumbel_player_cls=gumbel_player_cls,
        gumbel_mcts=gumbel_mcts,
        board_size=resolved["board_size"],
        seed=args.gumbel_seed,
        max_plies=args.max_plies,
    )

    frames = build_frames_v1(
        classic_snapshots=classic_snapshots,
        gumbel_snapshots=gumbel_snapshots,
        top_k=args.top_k,
        figsize=base.FIGSIZE,
        dpi=base.DPI,
        formula_font_size=args.formula_font_size,
    )

    output_path = base.REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_save:
        print(f"Built {len(frames)} dashboard frames. GIF encoding skipped.")
    else:
        base.save_gif(output_path, frames, duration=args.duration)
        print(f"Saved legacy dashboard GIF to {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metadata sidecar: {'yes' if metadata.get('checkpoint') else 'no'}")
    print(f"Text rendering: {'LaTeX' if use_tex else 'mathtext'}")
    print(f"Classic plies: {len(classic_snapshots)}")
    print(f"Gumbel plies: {len(gumbel_snapshots)}")
    print(f"Checkpoint iteration: {resolved['iteration']}")


if __name__ == "__main__":
    main()
