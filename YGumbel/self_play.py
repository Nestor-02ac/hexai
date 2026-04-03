"""
Batched self-play data generation for the Y Gumbel agent.
"""

from dataclasses import asdict, dataclass
import multiprocessing as mp
import time

import numpy as np
import torch

from config import GumbelZeroConfig
from y_board import Player, YBoard
from mcts import create_gumbel_mcts
from neural_net import YNet, encode_board
from progress import make_progress


class ReplayBuffer:
    """Circular buffer of (state, policy, value) tuples stored as numpy arrays."""

    def __init__(self, capacity, board_size):
        self.capacity = capacity
        n = board_size * (board_size + 1) // 2
        self.states = np.zeros((capacity, 4, board_size, board_size), dtype=np.float32)
        self.policies = np.zeros((capacity, n), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.index = 0

    def add(self, state, policy, value):
        self.states[self.index] = state
        self.policies[self.index] = policy
        self.values[self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_game(self, game_data):
        for state, policy, value in game_data:
            self.add(state, policy, value)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.policies[indices]),
            torch.from_numpy(self.values[indices]),
        )

    def __len__(self):
        return self.size


_WORKER_CONFIG = None
_WORKER_DEVICE = None
_WORKER_NETWORK = None


@dataclass
class SelfPlayProfile:
    search_seconds: float = 0.0
    prepare_seconds: float = 0.0
    forward_seconds: float = 0.0
    finish_seconds: float = 0.0
    finalize_seconds: float = 0.0
    store_seconds: float = 0.0
    root_prepares: int = 0
    leaf_prepares: int = 0
    eval_batches: int = 0
    evaluated_states: int = 0
    search_calls: int = 0
    commits: int = 0

    def merge(self, other):
        self.search_seconds += other.search_seconds
        self.prepare_seconds += other.prepare_seconds
        self.forward_seconds += other.forward_seconds
        self.finish_seconds += other.finish_seconds
        self.finalize_seconds += other.finalize_seconds
        self.store_seconds += other.store_seconds
        self.root_prepares += other.root_prepares
        self.leaf_prepares += other.leaf_prepares
        self.eval_batches += other.eval_batches
        self.evaluated_states += other.evaluated_states
        self.search_calls += other.search_calls
        self.commits += other.commits

    @property
    def profiled_seconds(self):
        return (
            self.search_seconds
            + self.prepare_seconds
            + self.forward_seconds
            + self.finish_seconds
            + self.finalize_seconds
            + self.store_seconds
        )


def _make_board(board_size):
    return YBoard(board_size)


def _init_self_play_worker(config_payload, state_dict, device_str):
    global _WORKER_CONFIG, _WORKER_DEVICE, _WORKER_NETWORK

    _WORKER_CONFIG = GumbelZeroConfig(**config_payload)
    _WORKER_CONFIG.device = device_str
    _WORKER_CONFIG.show_progress_bars = False
    _WORKER_CONFIG.num_self_play_workers = 1

    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_NETWORK = YNet(
        board_size=_WORKER_CONFIG.board_size,
        num_channels=_WORKER_CONFIG.num_channels,
        num_res_blocks=_WORKER_CONFIG.num_res_blocks,
    ).to(_WORKER_DEVICE)
    _WORKER_NETWORK.load_state_dict(state_dict)
    _WORKER_NETWORK.eval()
    torch.set_grad_enabled(False)


def _run_self_play_chunk(task):
    chunk_games, seed = task

    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(_WORKER_DEVICE).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    games, profile = _generate_self_play_data_serial(
        _WORKER_CONFIG,
        _WORKER_NETWORK,
        chunk_games,
        _WORKER_DEVICE,
        progress_total=None,
        return_profile=True,
    )
    return chunk_games, games, asdict(profile)


def _split_self_play_chunks(num_games, num_workers):
    target_chunks = min(num_games, max(num_workers, num_workers * 4))
    base = num_games // target_chunks
    remainder = num_games % target_chunks
    chunks = []
    for chunk_idx in range(target_chunks):
        chunk_size = base + (1 if chunk_idx < remainder else 0)
        if chunk_size > 0:
            chunks.append(chunk_size)
    return chunks


def generate_self_play_data(config, network, num_games, device, progress_total=None, return_profile=False):
    if config.num_self_play_workers <= 1 or num_games <= 1:
        return _generate_self_play_data_serial(
            config,
            network,
            num_games,
            device,
            progress_total=progress_total,
            return_profile=return_profile,
        )
    return _generate_self_play_data_parallel(config, network, num_games, device, return_profile=return_profile)


def _generate_self_play_data_parallel(config, network, num_games, device, return_profile=False):
    device_str = str(device)
    chunk_sizes = _split_self_play_chunks(num_games, config.num_self_play_workers)
    if len(chunk_sizes) <= 1:
        return _generate_self_play_data_serial(
            config,
            network,
            num_games,
            device,
            progress_total=None,
            return_profile=return_profile,
        )

    state_dict = {
        key: value.detach().cpu()
        for key, value in network.state_dict().items()
    }
    config_payload = asdict(config)
    base_seed = int(config.seed)

    progress = make_progress(
        total=num_games,
        desc="    self-play",
        unit="game",
        enabled=getattr(config, "show_progress_bars", True),
    )
    progress.set_postfix(workers=config.num_self_play_workers, chunks=f"0/{len(chunk_sizes)}")

    results = []
    profile = SelfPlayProfile()
    completed_chunks = 0
    ctx = mp.get_context("spawn")
    tasks = [(chunk_games, base_seed + chunk_idx) for chunk_idx, chunk_games in enumerate(chunk_sizes)]
    with ctx.Pool(
        processes=config.num_self_play_workers,
        initializer=_init_self_play_worker,
        initargs=(config_payload, state_dict, device_str),
    ) as pool:
        for chunk_games, games, profile_payload in pool.imap_unordered(_run_self_play_chunk, tasks):
            results.extend(games)
            if return_profile:
                profile.merge(SelfPlayProfile(**profile_payload))
            completed_chunks += 1
            progress.update(chunk_games)
            progress.set_postfix(
                workers=config.num_self_play_workers,
                chunks=f"{completed_chunks}/{len(chunk_sizes)}",
            )

    progress.close()
    if return_profile:
        return results, profile
    return results

def _generate_self_play_data_serial(config, network, num_games, device, progress_total=None, return_profile=False):
    """Generate self-play games with batched leaf evaluation."""
    network.eval()
    mcts = create_gumbel_mcts(config, network, device)
    profile = SelfPlayProfile() if return_profile else None
    estimated_total = max(int(progress_total), 1) if progress_total is not None else None
    progress = make_progress(
        total=estimated_total,
        desc="    self-play",
        unit="pos",
        enabled=getattr(config, "show_progress_bars", True),
    )

    boards = [_make_board(config.board_size) for _ in range(num_games)]
    currents = [Player.BLACK for _ in range(num_games)]
    searches = [None] * num_games
    trajectories = [[] for _ in range(num_games)]
    finished = [False] * num_games
    winners = [Player.EMPTY for _ in range(num_games)]
    positions_generated = [0]

    def _refresh_progress(force=False):
        if force or positions_generated[0] == 1 or positions_generated[0] % 8 == 0:
            finished_games = sum(1 if is_finished else 0 for is_finished in finished)
            active_games = sum(0 if is_finished else 1 for is_finished in finished)
            progress.set_postfix(done=f"{finished_games}/{num_games}", active=active_games)

    def _grow_total_if_needed():
        nonlocal estimated_total
        if estimated_total is None:
            return
        if positions_generated[0] < estimated_total:
            return
        estimated_total = max(estimated_total + max(32, estimated_total // 12), positions_generated[0] + 1)
        progress.set_total(estimated_total)

    def _start_turn(game_idx):
        searches[game_idx] = mcts.new_search(int(currents[game_idx]))

    def _commit_move(game_idx):
        player = int(currents[game_idx])
        if profile is not None:
            t_store = time.perf_counter()
        state_np = encode_board(boards[game_idx], player).numpy()
        if profile is not None:
            profile.store_seconds += time.perf_counter() - t_store
            t_finalize = time.perf_counter()
        action, policy = mcts.finalize_search(searches[game_idx])
        if profile is not None:
            profile.finalize_seconds += time.perf_counter() - t_finalize
            profile.commits += 1
        trajectories[game_idx].append((state_np, policy, player))
        positions_generated[0] += 1
        _grow_total_if_needed()
        progress.update(1)
        _refresh_progress()

        boards[game_idx].play(action, currents[game_idx])
        if boards[game_idx].check_win(currents[game_idx]):
            winners[game_idx] = currents[game_idx]
            finished[game_idx] = True
            _refresh_progress(force=True)
            return

        currents[game_idx] = Player.WHITE if currents[game_idx] == Player.BLACK else Player.BLACK
        _start_turn(game_idx)

    for game_idx in range(num_games):
        _start_turn(game_idx)

    while not all(finished):
        eval_requests = []

        for game_idx in range(num_games):
            if finished[game_idx]:
                continue

            search = searches[game_idx]
            if search.root.count == 0:
                if profile is not None:
                    t_prepare = time.perf_counter()
                state, legal_actions = mcts.prepare_expand(
                    search.root,
                    boards[game_idx],
                    int(currents[game_idx]),
                )
                if profile is not None:
                    profile.prepare_seconds += time.perf_counter() - t_prepare
                    profile.root_prepares += 1
                eval_requests.append(("root", game_idx, state, legal_actions, None))
                continue

            if mcts.search_complete(search):
                _commit_move(game_idx)
                continue

            if profile is not None:
                t_search = time.perf_counter()
            leaf_request = mcts.simulate_until_leaf(search, boards[game_idx])
            if profile is not None:
                profile.search_seconds += time.perf_counter() - t_search
                profile.search_calls += 1
            if leaf_request is None:
                if mcts.search_complete(search):
                    _commit_move(game_idx)
                continue

            if profile is not None:
                t_prepare = time.perf_counter()
            state, legal_actions = mcts.prepare_expand(
                leaf_request.node,
                leaf_request.board,
                leaf_request.to_play,
            )
            if profile is not None:
                profile.prepare_seconds += time.perf_counter() - t_prepare
                profile.leaf_prepares += 1
            eval_requests.append(("leaf", game_idx, state, legal_actions, leaf_request))

        if not eval_requests:
            continue

        if profile is not None:
            t_forward = time.perf_counter()
        policy_batch, value_batch = mcts.evaluate_states([request[2] for request in eval_requests])
        if profile is not None:
            profile.forward_seconds += time.perf_counter() - t_forward
            profile.eval_batches += 1
            profile.evaluated_states += len(eval_requests)
        for idx, (request_type, game_idx, _state, legal_actions, extra) in enumerate(eval_requests):
            policy_logits = policy_batch[idx]
            value = float(value_batch[idx])
            if profile is not None:
                t_finish = time.perf_counter()
            if request_type == "root":
                mcts.finish_root(
                    searches[game_idx],
                    policy_logits,
                    value,
                    legal_actions,
                    add_noise=config.use_gumbel_noise,
                )
            else:
                mcts.finish_leaf(extra, policy_logits, value, legal_actions)
            if profile is not None:
                profile.finish_seconds += time.perf_counter() - t_finish

    all_data = []
    for game_idx in range(num_games):
        winner = int(winners[game_idx])
        outcome = 1.0 if winner == Player.BLACK else -1.0
        game_data = []
        for state_np, policy_np, _player_at_pos in trajectories[game_idx]:
            game_data.append((state_np, policy_np, outcome))
        all_data.append(game_data)

    progress.set_total(positions_generated[0])
    _refresh_progress(force=True)
    progress.close()
    if return_profile:
        return all_data, profile
    return all_data


def play_self_play_game(config, network, device):
    return generate_self_play_data(config, network, 1, device)[0]