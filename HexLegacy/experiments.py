"""
Experimental verification of results from "Monte-Carlo Hex" (Cazenave & Saffidine).

Reproduces (partially) the experimental results from the paper:
  Table 1: Effect of simulation count vs 16000 reference
  Table 2: UCT constant variation with RAVE
  Table 3: Random vs bridge-based simulations (subset of paper's 4 types)
  Table 4: RAVE bias variation

Limitations vs the paper:
  - Table 3 only compares type1 (random) vs type2 (bridges). The paper
    also tests type3 (bridges + level-2 templates) and type4 (+ Ziggurats),
    which are not implemented.
  - The paper's section 2.3 experiment (virtual connection solver integration,
    69.5% result) is not reproduced as it requires Anshelevich's VC algorithm.
  - The paper's YOPT is written in C/C++ and much faster per simulation.
    Our Python implementation is ~100x slower, so full 11x11 experiments
    with 16000+ simulations are extremely slow.

Usage:
  python experiments.py sanity       # Quick test: MCTS vs Random (seconds)
  python experiments.py small        # Scaled-down trend verification (minutes)
  python experiments.py table1       # Full Table 1 at 11x11 (hours)
  python experiments.py table2       # Full Table 2 at 11x11 (hours)
  python experiments.py table3       # Full Table 3 at 11x11 (hours)
  python experiments.py table4       # Full Table 4 at 11x11 (hours)
  python experiments.py all          # All tables (many hours)

  Optional: python experiments.py <mode> --seed 42   # For reproducibility
"""

import time
import random
import sys
import json
import os
import multiprocessing
from datetime import datetime

from hex_board import HexBoard, Player
from mcts_hex import MCTSHex, SimulationType, play_game, RandomAgent

# Number of parallel workers (default: all CPU cores)
NUM_WORKERS = multiprocessing.cpu_count()


def _play_single_game(args):
    """Worker function for parallel game execution. Must be top-level for pickling."""
    game_idx, size, agent1_params, agent2_params, game_seed = args
    random.seed(game_seed)

    agent1 = MCTSHex(board_size=size, **agent1_params)
    agent2 = MCTSHex(board_size=size, **agent2_params)

    if game_idx % 2 == 0:
        winner = play_game(size, black_agent=agent1, white_agent=agent2)
        agent1_won = (winner == Player.BLACK)
    else:
        winner = play_game(size, black_agent=agent2, white_agent=agent1)
        agent1_won = (winner == Player.WHITE)

    return game_idx, 1 if agent1_won else 0


def _play_single_game_vs_random(args):
    """Worker function for MCTS vs Random games."""
    game_idx, size, mcts_params, game_seed = args
    random.seed(game_seed)

    mcts = MCTSHex(board_size=size, **mcts_params)
    rand_agent = RandomAgent()

    if game_idx % 2 == 0:
        winner = play_game(size, black_agent=mcts, white_agent=rand_agent)
        mcts_won = (winner == Player.BLACK)
    else:
        winner = play_game(size, black_agent=rand_agent, white_agent=mcts)
        mcts_won = (winner == Player.WHITE)

    return game_idx, 1 if mcts_won else 0

# Directory for saving results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_json_serializable(obj):
    """Convert dict keys to strings for JSON compatibility."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    return obj


def save_results(experiment_name, data, seed=None):
    """
    Save experiment results to a JSON file in results/.
    File: results/<experiment_name>_<timestamp>.json
    Also appends to results/all_results.jsonl for easy aggregation.
    """
    _ensure_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "data": _make_json_serializable(data),
    }

    # Save individual JSON file
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Results saved to: {filepath}")

    # Append to JSONL log (one record per line, easy to grep/aggregate)
    log_path = os.path.join(RESULTS_DIR, "all_results.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Appended to: {log_path}")

    return filepath


def run_experiment(size, agent1_params, agent2_params, num_games=200, desc="",
                   num_workers=None):
    """
    Run a match: num_games/2 as Black, num_games/2 as White.
    Games run in parallel across CPU cores.
    Returns dict with win_pct, wins, num_games, elapsed_s, game_results, and params.
    """
    if num_workers is None:
        num_workers = NUM_WORKERS

    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Agent1: {agent1_params}")
    print(f"  Agent2: {agent2_params}")
    print(f"  Games: {num_games}, Board: {size}x{size}, Workers: {num_workers}")
    print(f"{'='*60}")

    start = time.time()

    # Generate unique seed per game from parent RNG
    game_seeds = [random.randrange(2**31) for _ in range(num_games)]
    tasks = [(i, size, agent1_params, agent2_params, game_seeds[i])
             for i in range(num_games)]

    game_results = [0] * num_games
    wins = 0

    if num_workers <= 1:
        # Sequential fallback
        for i, task in enumerate(tasks):
            _, result = _play_single_game(task)
            game_results[i] = result
            wins += result
            elapsed = time.time() - start
            if (i + 1) % max(1, num_games // 10) == 0 or (i + 1) == num_games:
                pct = 100.0 * wins / (i + 1)
                print(f"  [{i+1}/{num_games}] {wins} wins = {pct:.1f}% ({elapsed:.1f}s)")
    else:
        # Parallel execution
        completed = 0
        with multiprocessing.Pool(num_workers) as pool:
            for game_idx, result in pool.imap_unordered(_play_single_game, tasks):
                game_results[game_idx] = result
                wins += result
                completed += 1
                elapsed = time.time() - start
                if completed % max(1, num_games // 10) == 0 or completed == num_games:
                    pct = 100.0 * wins / completed
                    print(f"  [{completed}/{num_games}] {wins} wins = {pct:.1f}% ({elapsed:.1f}s)")

    pct = 100.0 * wins / num_games
    total_time = time.time() - start
    print(f"  RESULT: {wins}/{num_games} = {pct:.1f}% ({total_time:.1f}s)")

    return {
        "description": desc,
        "board_size": size,
        "num_games": num_games,
        "wins": wins,
        "win_pct": round(pct, 2),
        "elapsed_s": round(total_time, 2),
        "num_workers": num_workers,
        "agent1_params": agent1_params,
        "agent2_params": agent2_params,
        "game_results": game_results,
    }


def quick_sanity_check(seed=None):
    """Quick test: MCTS should dominate Random."""
    print("\n" + "#" * 60)
    print("  SANITY CHECK: MCTS (500 sims) vs Random on 5x5")
    print("#" * 60)

    size = 5
    num_games = 20

    mcts_params = {
        'num_simulations': 500,
        'use_rave': True,
        'c_uct': 0.0,
        'rave_bias': 0.00025,
        'simulation_type': SimulationType.BRIDGES,
    }

    wins = 0
    game_results = [0] * num_games
    start = time.time()

    game_seeds = [random.randrange(2**31) for _ in range(num_games)]
    tasks = [(i, size, mcts_params, game_seeds[i]) for i in range(num_games)]

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        for game_idx, result in pool.imap_unordered(_play_single_game_vs_random, tasks):
            game_results[game_idx] = result
            wins += result
            print(f"  Game {game_idx+1}: MCTS {'WIN' if result else 'LOSS'}")

    elapsed = time.time() - start
    pct = 100.0 * wins / num_games
    print(f"\n  MCTS wins: {wins}/{num_games} = {pct:.1f}% ({elapsed:.1f}s)")
    print(f"  Expected: >80%")

    data = {
        "board_size": size,
        "num_games": num_games,
        "wins": wins,
        "win_pct": round(pct, 2),
        "elapsed_s": round(elapsed, 2),
        "mcts_params": mcts_params,
        "opponent": "random",
        "game_results": game_results,
    }
    save_results("sanity_check", data, seed=seed)
    return data


def small_experiment(seed=None):
    """
    Scaled-down experiments on 5x5 board with fewer sims/games.
    These verify the same TRENDS as the paper, not exact numbers.
    5x5 board != 11x11, and 500 sims != 16000 sims.
    """
    print("\n" + "=" * 60)
    print("  SMALL-SCALE TREND VERIFICATION (5x5, 500 sims, 30 games)")
    print("  These verify trends, NOT exact paper numbers.")
    print("=" * 60)

    size = 5
    ng = 30
    all_data = {"board_size": size, "num_games_per_exp": ng, "experiments": {}}

    # --- Mini Table 1: More simulations = stronger ---
    print("\n" + "-" * 50)
    print("  Mini Table 1: Simulation count effect")
    print("  Paper trend: fewer sims < 50%, more sims > 50%")
    print("-" * 50)
    ref = {
        'num_simulations': 500, 'use_rave': True, 'c_uct': 0.0,
        'rave_bias': 0.00025, 'simulation_type': SimulationType.BRIDGES,
    }
    r1 = {}
    for sims in [100, 200, 1000, 2000]:
        t = {**ref, 'num_simulations': sims}
        result = run_experiment(size, t, ref, ng,
                                desc=f"Mini T1: {sims} vs 500 sims")
        r1[sims] = result
    print(f"\n  Mini Table 1 Summary (vs 500 ref):")
    for s, r in sorted(r1.items()):
        marker = "<50%" if r["win_pct"] < 50 else ">50%"
        print(f"    {s:>5} sims: {r['win_pct']:.1f}% ({marker})")
    all_data["experiments"]["mini_table1_sim_count"] = {
        "reference_sims": 500,
        "results": {str(k): v for k, v in r1.items()},
    }

    # --- Mini Table 2: UCT constant ---
    print("\n" + "-" * 50)
    print("  Mini Table 2: UCT constant with RAVE")
    print("  Paper trend: C=0 best, higher C = worse")
    print("-" * 50)
    ref2 = {
        'num_simulations': 500, 'use_rave': True, 'c_uct': 0.3,
        'rave_bias': 0.00025, 'simulation_type': SimulationType.BRIDGES,
    }
    r2 = {}
    for c in [0.0, 0.1, 0.5, 0.7]:
        t = {**ref2, 'c_uct': c}
        result = run_experiment(size, t, ref2, ng,
                                desc=f"Mini T2: C={c} vs C=0.3")
        r2[c] = result
    print(f"\n  Mini Table 2 Summary (vs C=0.3 ref):")
    for c, r in sorted(r2.items()):
        print(f"    C={c:.1f}: {r['win_pct']:.1f}%")
    all_data["experiments"]["mini_table2_uct_constant"] = {
        "reference_c_uct": 0.3,
        "results": {str(k): v for k, v in r2.items()},
    }

    # --- Mini Table 3: Random vs Bridges ---
    print("\n" + "-" * 50)
    print("  Mini Table 3: Random vs Bridge simulations")
    print("  Paper: type1 (random) gets 22% vs type3 (bridges+level2)")
    print("  We test: random vs bridges-only")
    print("-" * 50)
    ref3 = {
        'num_simulations': 500, 'use_rave': True, 'c_uct': 0.0,
        'rave_bias': 0.00025, 'simulation_type': SimulationType.BRIDGES,
    }
    t3 = {**ref3, 'simulation_type': SimulationType.RANDOM}
    r3 = run_experiment(size, t3, ref3, ng,
                         desc="Mini T3: Random vs Bridges")
    print(f"\n  Random vs Bridges: {r3['win_pct']:.1f}%")
    print(f"  Paper: random gets 22% vs bridges+templates (expect <50%)")
    all_data["experiments"]["mini_table3_templates"] = {
        "reference": "bridges",
        "results": {"random_vs_bridges": r3},
    }

    # --- RAVE vs no-RAVE ---
    print("\n" + "-" * 50)
    print("  RAVE vs Pure UCT")
    print("  Paper: RAVE significantly improves play")
    print("-" * 50)
    rave_p = {
        'num_simulations': 500, 'use_rave': True, 'c_uct': 0.0,
        'rave_bias': 0.00025, 'simulation_type': SimulationType.BRIDGES,
    }
    no_rave_p = {
        'num_simulations': 500, 'use_rave': False, 'c_uct': 0.5,
        'simulation_type': SimulationType.BRIDGES,
    }
    r_rave = run_experiment(size, rave_p, no_rave_p, ng,
                             desc="RAVE (C=0) vs Pure UCT (C=0.5)")
    print(f"\n  RAVE vs no-RAVE: {r_rave['win_pct']:.1f}% (expected >55%)")
    all_data["experiments"]["rave_vs_pure_uct"] = r_rave

    save_results("small_experiment", all_data, seed=seed)
    return all_data


# ===== Full-scale paper reproductions (11x11, 200 games) =====

def table1_simulations(size=11, num_games=200, seed=None):
    """
    Table 1: Increasing simulations improves level.
    Reference: 16000 simulations with RAVE, UCT=0.3, bridges.

    Paper results (200 games, 100 as Black + 100 as White):
      1000:  6%     2000: 11.5%
      4000: 20%     8000: 33%
     32000: 61%    64000: 68.5%
    """
    print("\n" + "#" * 60)
    print("  TABLE 1: Increasing simulations improves the level")
    print("  Reference: 16000 sims, RAVE, C=0.3, bridges")
    print("#" * 60)

    ref = {
        'num_simulations': 16000, 'use_rave': True,
        'c_uct': 0.3, 'rave_bias': 0.00025,
        'simulation_type': SimulationType.BRIDGES,
    }
    paper = {1000: 6, 2000: 11.5, 4000: 20, 8000: 33, 32000: 61, 64000: 68.5}
    results = {}

    for sims in [1000, 2000, 4000, 8000, 32000, 64000]:
        t = {**ref, 'num_simulations': sims}
        results[sims] = run_experiment(size, t, ref, num_games,
                                        desc=f"T1: {sims} vs 16000")

    print(f"\n  TABLE 1 SUMMARY:")
    print(f"  {'Sims':>8} | {'Ours':>7} | {'Paper':>7}")
    print(f"  {'-'*28}")
    for s in sorted(results):
        print(f"  {s:>8} | {results[s]['win_pct']:>6.1f}% | {paper[s]:>6.1f}%")

    data = {
        "board_size": size, "num_games": num_games,
        "reference_params": ref,
        "paper_results": {str(k): v for k, v in paper.items()},
        "our_results": {str(k): v for k, v in results.items()},
    }
    save_results("table1_simulations", data, seed=seed)
    return data


def table2_uct_constant(size=11, num_games=200, seed=None):
    """
    Table 2: Variation of UCT constant with RAVE.
    Reference: UCT=0.3, 16000 sims, RAVE.

    Paper results:
      0.0: 61%    0.1: 60%    0.2: 55.5%
      0.4: 42%    0.5: 41%    0.6: 35.5%    0.7: 32.5%
    """
    print("\n" + "#" * 60)
    print("  TABLE 2: Variation of the UCT constant")
    print("  Reference: C=0.3, 16000 sims, RAVE, bridges")
    print("#" * 60)

    base = {
        'num_simulations': 16000, 'use_rave': True,
        'rave_bias': 0.00025, 'simulation_type': SimulationType.BRIDGES,
    }
    ref = {**base, 'c_uct': 0.3}
    paper = {0.0: 61, 0.1: 60, 0.2: 55.5, 0.4: 42, 0.5: 41, 0.6: 35.5, 0.7: 32.5}
    results = {}

    for c in [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7]:
        t = {**base, 'c_uct': c}
        results[c] = run_experiment(size, t, ref, num_games,
                                     desc=f"T2: C={c} vs C=0.3")

    print(f"\n  TABLE 2 SUMMARY:")
    print(f"  {'C':>6} | {'Ours':>7} | {'Paper':>7}")
    print(f"  {'-'*25}")
    for c in [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7]:
        print(f"  {c:>6.1f} | {results[c]['win_pct']:>6.1f}% | {paper[c]:>6.1f}%")

    data = {
        "board_size": size, "num_games": num_games,
        "reference_c_uct": 0.3,
        "paper_results": {str(k): v for k, v in paper.items()},
        "our_results": {str(k): v for k, v in results.items()},
    }
    save_results("table2_uct_constant", data, seed=seed)
    return data


def table3_templates(size=11, num_games=200, seed=None):
    """
    Table 3: Using templates during simulations.
    Reference: type3 (bridges + level-2 templates) — we use type2 (bridges only)
    as a proxy since level-2 templates are not implemented.

    Paper results (vs type3):
      type1 (random):                    22%
      type2 (bridges):                   42%
      type4 (bridges+level2+ziggurats):  71.5%

    We can only test type1 vs type2. The paper's reference is type3 which
    is stronger than our type2, so our random-vs-bridges gap will be
    narrower than the paper's random-vs-type3 gap.
    """
    print("\n" + "#" * 60)
    print("  TABLE 3: Templates during simulations (partial)")
    print("  We test: random vs bridges (paper tests 4 types)")
    print("  Paper's reference is bridges+level2 (not implemented)")
    print("#" * 60)

    base = {
        'num_simulations': 16000, 'use_rave': True,
        'c_uct': 0.0, 'rave_bias': 0.00025,
    }
    ref = {**base, 'simulation_type': SimulationType.BRIDGES}
    t1 = {**base, 'simulation_type': SimulationType.RANDOM}
    r1 = run_experiment(size, t1, ref, num_games,
                         desc="T3: type1 (random) vs type2 (bridges)")

    print(f"\n  TABLE 3 SUMMARY:")
    print(f"  {'Type':>10} | {'Ours':>7} | {'Paper (vs type3)':>16}")
    print(f"  {'-'*40}")
    print(f"  {'random':>10} | {r1['win_pct']:>6.1f}% | {'22.0%':>16}")
    print(f"  Note: paper reference is type3 (bridges+level2),")
    print(f"        ours is type2 (bridges only), so expect higher %.")

    data = {
        "board_size": size, "num_games": num_games,
        "reference": "bridges (type2)",
        "paper_results": {"type1_random": 22.0, "type2_bridges": 42.0, "type4_ziggurats": 71.5},
        "our_results": {"random_vs_bridges": r1},
    }
    save_results("table3_templates", data, seed=seed)
    return data


def table4_rave_bias(size=11, num_games=200, seed=None):
    """
    Table 4: RAVE bias variation.
    Reference: bias=0.001.

    Paper results:
      0.0005:   50.5%
      0.00025:  59%
      0.000125: 53.5%
    """
    print("\n" + "#" * 60)
    print("  TABLE 4: RAVE bias")
    print("  Reference: bias=0.001, 16000 sims, RAVE, bridges")
    print("#" * 60)

    base = {
        'num_simulations': 16000, 'use_rave': True,
        'c_uct': 0.0, 'simulation_type': SimulationType.BRIDGES,
    }
    ref = {**base, 'rave_bias': 0.001}
    paper = {0.0005: 50.5, 0.00025: 59, 0.000125: 53.5}
    results = {}

    for bias in [0.0005, 0.00025, 0.000125]:
        t = {**base, 'rave_bias': bias}
        results[bias] = run_experiment(size, t, ref, num_games,
                                        desc=f"T4: bias={bias} vs 0.001")

    print(f"\n  TABLE 4 SUMMARY:")
    print(f"  {'Bias':>10} | {'Ours':>7} | {'Paper':>7}")
    print(f"  {'-'*30}")
    for b in [0.0005, 0.00025, 0.000125]:
        print(f"  {b:>10.6f} | {results[b]['win_pct']:>6.1f}% | {paper[b]:>6.1f}%")

    data = {
        "board_size": size, "num_games": num_games,
        "reference_bias": 0.001,
        "paper_results": {str(k): v for k, v in paper.items()},
        "our_results": {str(k): v for k, v in results.items()},
    }
    save_results("table4_rave_bias", data, seed=seed)
    return data


if __name__ == '__main__':
    args = sys.argv[1:]

    # Parse optional --seed flag
    seed = None
    if '--seed' in args:
        idx = args.index('--seed')
        if idx + 1 < len(args):
            seed = int(args[idx + 1])
            args = args[:idx] + args[idx+2:]
        else:
            print("Error: --seed requires a value")
            sys.exit(1)

    # Parse optional --workers flag
    if '--workers' in args:
        idx = args.index('--workers')
        if idx + 1 < len(args):
            NUM_WORKERS = int(args[idx + 1])
            args = args[:idx] + args[idx+2:]
        else:
            print("Error: --workers requires a value")
            sys.exit(1)

    if seed is not None:
        random.seed(seed)
        print(f"  Random seed set to: {seed}")

    print(f"  Using {NUM_WORKERS} parallel workers")

    mode = args[0] if args else 'default'

    if mode == 'sanity':
        quick_sanity_check(seed=seed)
    elif mode == 'small':
        small_experiment(seed=seed)
    elif mode == 'table1':
        table1_simulations(seed=seed)
    elif mode == 'table2':
        table2_uct_constant(seed=seed)
    elif mode == 'table3':
        table3_templates(seed=seed)
    elif mode == 'table4':
        table4_rave_bias(seed=seed)
    elif mode == 'all':
        table1_simulations(seed=seed)
        table2_uct_constant(seed=seed)
        table3_templates(seed=seed)
        table4_rave_bias(seed=seed)
    elif mode == 'default':
        quick_sanity_check(seed=seed)
        small_experiment(seed=seed)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python experiments.py [sanity|small|table1|table2|table3|table4|all] [--seed N] [--workers N]")
        sys.exit(1)
