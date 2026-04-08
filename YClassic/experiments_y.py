"""
Comparative experiments for YClassic.

Runs the same protocol shape as HexClassic, adapted to Y:
  Table 1: Effect of simulation count vs 16000 reference
  Table 2: UCT constant variation with RAVE
  Table 3: Random vs connectivity-based simulations
  Table 4: RAVE bias variation

Two backends are available:
  - Python (mcts_y.py): pure Python reference implementation.
  - Cython (cmcts_y.pyx): fast backend, enable with --cython.
Both parallelize at game level via multiprocessing.

Unlike Hex, Y does not have a direct bridge-template analogue, so the stronger
rollout policy is a connectivity heuristic rather than a bridge-defense rule.
"""

import json
import multiprocessing
import os
import random
import sys
import time
from datetime import datetime

try:
    from y_board import Player
    from mcts_y import MCTSY, RandomAgent, SimulationType, play_game
except ModuleNotFoundError:
    from .y_board import Player
    from .mcts_y import MCTSY, RandomAgent, SimulationType, play_game


NUM_WORKERS = multiprocessing.cpu_count()
USE_CYTHON = False
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _get_backend():
    if USE_CYTHON:
        from cmcts_y import CMCTSY, RandomAgent as CyRandom, play_game as cy_play
        return CMCTSY, cy_play, CyRandom
    return MCTSY, play_game, RandomAgent


def _seed_all(game_seed):
    random.seed(game_seed)
    if USE_CYTHON:
        from cmcts_y import seed_rng
        seed_rng(game_seed)


def _play_single_game(args):
    game_idx, size, agent1_params, agent2_params, game_seed = args
    _seed_all(game_seed)

    AgentClass, play_fn, _ = _get_backend()
    agent1 = AgentClass(board_size=size, **agent1_params)
    agent2 = AgentClass(board_size=size, **agent2_params)

    if game_idx % 2 == 0:
        winner = play_fn(size, black_agent=agent1, white_agent=agent2)
        agent1_won = (winner == Player.BLACK)
    else:
        winner = play_fn(size, black_agent=agent2, white_agent=agent1)
        agent1_won = (winner == Player.WHITE)

    return game_idx, 1 if agent1_won else 0


def _play_single_game_vs_random(args):
    game_idx, size, mcts_params, game_seed = args
    _seed_all(game_seed)

    AgentClass, play_fn, RandClass = _get_backend()
    mcts = AgentClass(board_size=size, **mcts_params)
    rand_agent = RandClass()

    if game_idx % 2 == 0:
        winner = play_fn(size, black_agent=mcts, white_agent=rand_agent)
        mcts_won = (winner == Player.BLACK)
    else:
        winner = play_fn(size, black_agent=rand_agent, white_agent=mcts)
        mcts_won = (winner == Player.WHITE)

    return game_idx, 1 if mcts_won else 0


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    return obj


def save_results(experiment_name, data, seed=None):
    """
    Save experiment results to a JSON file in results/.
    File: results/<experiment_name>_<timestamp>.json
    Also append to results/all_results.jsonl.
    """
    _ensure_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "data": _make_json_serializable(data),
    }

    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Results saved to: {filepath}")

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

    print()
    print(f"  {desc}")
    print(f"  Agent1: {agent1_params}")
    print(f"  Agent2: {agent2_params}")
    print(f"  Games: {num_games}, Board: side length {size}, Workers: {num_workers}")

    start = time.time()
    game_seeds = [random.randrange(2**31) for _ in range(num_games)]
    tasks = [(i, size, agent1_params, agent2_params, game_seeds[i])
             for i in range(num_games)]

    game_results = [0] * num_games
    wins = 0

    if num_workers <= 1:
        for i, task in enumerate(tasks):
            _, result = _play_single_game(task)
            game_results[i] = result
            wins += result
            elapsed = time.time() - start
            if (i + 1) % max(1, num_games // 10) == 0 or (i + 1) == num_games:
                pct = 100.0 * wins / (i + 1)
                print(f"  [{i+1}/{num_games}] {wins} wins = {pct:.1f}% ({elapsed:.1f}s)")
    else:
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
    print("\nSanity check: MCTS (500 sims) vs Random on size 5")

    size = 5
    num_games = 20
    mcts_params = {
        "num_simulations": 500,
        "use_rave": True,
        "c_uct": 0.0,
        "rave_bias": 0.00025,
        "simulation_type": SimulationType.CONNECTIVITY,
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
    print("  Expected: >80%")

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
    Scaled-down experiments on 7-side Y with fewer sims/games.
    Quick way to verify trends before the full 11-side runs.
    """
    print()
    print("  Small-scale trend verification (side 7, 1000 sims, 30 games)")

    size = 7
    ng = 30
    all_data = {"board_size": size, "num_games_per_exp": ng, "experiments": {}}

    ref = {
        "num_simulations": 1000, "use_rave": True, "c_uct": 0.0,
        "rave_bias": 0.00025, "simulation_type": SimulationType.CONNECTIVITY,
    }

    print("\n  Mini Table 1: simulation count")
    r1 = {}
    for sims in [250, 500, 2000, 4000]:
        t = {**ref, "num_simulations": sims}
        result = run_experiment(size, t, ref, ng,
                                desc=f"Mini T1: {sims} vs 1000 sims")
        r1[sims] = result
    all_data["experiments"]["mini_table1_sim_count"] = {
        "reference_sims": 1000,
        "results": {str(k): v for k, v in r1.items()},
    }

    print("\n  Mini Table 2: UCT constant with RAVE")
    ref2 = {
        "num_simulations": 1000, "use_rave": True, "c_uct": 0.3,
        "rave_bias": 0.00025, "simulation_type": SimulationType.CONNECTIVITY,
    }
    r2 = {}
    for c in [0.0, 0.1, 0.5, 0.7]:
        t = {**ref2, "c_uct": c}
        result = run_experiment(size, t, ref2, ng,
                                desc=f"Mini T2: C={c} vs C=0.3")
        r2[c] = result
    all_data["experiments"]["mini_table2_uct_constant"] = {
        "reference_c_uct": 0.3,
        "results": {str(k): v for k, v in r2.items()},
    }

    print("\n  Mini Table 3: random vs connectivity rollouts")
    ref3 = {
        "num_simulations": 1000, "use_rave": True, "c_uct": 0.0,
        "rave_bias": 0.00025, "simulation_type": SimulationType.CONNECTIVITY,
    }
    t3 = {**ref3, "simulation_type": SimulationType.RANDOM}
    r3 = run_experiment(size, t3, ref3, ng,
                        desc="Mini T3: Random vs Connectivity")
    all_data["experiments"]["mini_table3_rollout_policy"] = {
        "reference": "connectivity",
        "results": {"random_vs_connectivity": r3},
    }

    print("\n  Mini Table 4: RAVE bias")
    ref4 = {
        "num_simulations": 1000, "use_rave": True, "c_uct": 0.0,
        "rave_bias": 0.001, "simulation_type": SimulationType.CONNECTIVITY,
    }
    r4 = {}
    for b in [0.0005, 0.00025, 0.000125]:
        t = {**ref4, "rave_bias": b}
        result = run_experiment(size, t, ref4, ng,
                                desc=f"Mini T4: bias={b} vs bias=0.001")
        r4[b] = result
    all_data["experiments"]["mini_table4_rave_bias"] = {
        "reference_bias": 0.001,
        "results": {str(k): v for k, v in r4.items()},
    }

    save_results("small", all_data, seed=seed)
    return all_data


def experiment_table1(seed=None, num_workers=None):
    print("\nTable 1: simulation count")

    size = 11
    ref = {
        "num_simulations": 16000,
        "use_rave": True,
        "c_uct": 0.0,
        "rave_bias": 0.00025,
        "simulation_type": SimulationType.CONNECTIVITY,
    }
    num_games = 200
    results = {}

    for sims in [1000, 2000, 4000, 8000, 32000, 64000]:
        test = {**ref, "num_simulations": sims}
        result = run_experiment(size, test, ref, num_games,
                                f"Table 1: {sims} sims vs 16000 sims",
                                num_workers)
        results[sims] = result

    save_results("table1_simulations", results, seed)
    return results


def experiment_table2(seed=None, num_workers=None):
    print("\nTable 2: UCT exploration constant")

    size = 11
    base = {
        "num_simulations": 16000,
        "use_rave": True,
        "rave_bias": 0.00025,
        "simulation_type": SimulationType.CONNECTIVITY,
    }
    ref = {**base, "c_uct": 0.3}
    num_games = 200
    results = {}

    for c in [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7]:
        test = {**base, "c_uct": c}
        result = run_experiment(size, test, ref, num_games,
                                f"Table 2: C={c} vs C=0.3",
                                num_workers)
        results[c] = result

    save_results("table2_uct_constant", results, seed)
    return results


def experiment_table3(seed=None, num_workers=None):
    print("\nTable 3: rollout policy")

    size = 11
    ref = {
        "num_simulations": 16000,
        "use_rave": True,
        "c_uct": 0.0,
        "rave_bias": 0.00025,
        "simulation_type": SimulationType.CONNECTIVITY,
    }
    test = {**ref, "simulation_type": SimulationType.RANDOM}
    num_games = 200

    result = run_experiment(size, test, ref, num_games,
                            "Table 3: Random vs Connectivity",
                            num_workers)
    save_results("table3_rollout_policy", result, seed)
    return result


def experiment_table4(seed=None, num_workers=None):
    print("\nTable 4: RAVE bias")

    size = 11
    base = {
        "num_simulations": 16000,
        "use_rave": True,
        "c_uct": 0.0,
        "simulation_type": SimulationType.CONNECTIVITY,
    }
    ref = {**base, "rave_bias": 0.001}
    num_games = 200
    results = {}

    for b in [0.0005, 0.00025, 0.000125]:
        test = {**base, "rave_bias": b}
        result = run_experiment(size, test, ref, num_games,
                                f"Table 4: bias={b} vs bias=0.001",
                                num_workers)
        results[b] = result

    save_results("table4_rave_bias", results, seed)
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "sanity"

    seed = None
    num_workers = NUM_WORKERS

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--seed":
            i += 1
            seed = int(args[i])
        elif arg == "--workers":
            i += 1
            num_workers = int(args[i])
        elif arg == "--cython":
            USE_CYTHON = True
        elif arg.startswith("-"):
            print(f"Unknown option: {arg}")
            sys.exit(1)
        else:
            mode = arg
        i += 1

    if seed is not None:
        random.seed(seed)
        print(f"Using seed = {seed}")

    print(f"Workers = {num_workers}")
    print(f"Backend = {'Cython' if USE_CYTHON else 'Python'}")

    if mode == "sanity":
        quick_sanity_check(seed)
    elif mode == "small":
        small_experiment(seed)
    elif mode == "table1":
        experiment_table1(seed, num_workers)
    elif mode == "table2":
        experiment_table2(seed, num_workers)
    elif mode == "table3":
        experiment_table3(seed, num_workers)
    elif mode == "table4":
        experiment_table4(seed, num_workers)
    elif mode == "all":
        experiment_table1(seed, num_workers)
        experiment_table2(seed, num_workers)
        experiment_table3(seed, num_workers)
        experiment_table4(seed, num_workers)
    else:
        print("Usage: python experiments_y.py [sanity|small|table1|table2|table3|table4|all] [--seed N] [--workers N] [--cython]")
        sys.exit(1)
