"""Batched self-play data generation for the Hex Gumbel agent."""

import numpy as np
import torch

from hex_board import Player
from mcts import GumbelMCTS
from neural_net import encode_board


class ReplayBuffer:
    """Circular buffer of (state, policy, value) tuples stored as numpy arrays."""

    def __init__(self, capacity, board_size):
        self.capacity = capacity
        n = board_size * board_size
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


def _make_board(board_size):
    try:
        from chex_board import CHexBoard
        return CHexBoard(board_size)
    except ImportError:
        from hex_board import HexBoard
        return HexBoard(board_size)


def generate_self_play_data(config, network, num_games, device):
    """Generate self-play games with batched leaf evaluation."""
    network.eval()
    mcts = GumbelMCTS(config, network, device)

    boards = [_make_board(config.board_size) for _ in range(num_games)]
    currents = [Player.BLACK for _ in range(num_games)]
    searches = [None] * num_games
    trajectories = [[] for _ in range(num_games)]
    finished = [False] * num_games
    winners = [Player.EMPTY for _ in range(num_games)]

    def _start_turn(game_idx):
        searches[game_idx] = mcts.new_search(int(currents[game_idx]))

    def _commit_move(game_idx):
        player = int(currents[game_idx])
        state_np = encode_board(boards[game_idx], player).numpy()
        action, policy = mcts.finalize_search(searches[game_idx])
        trajectories[game_idx].append((state_np, policy, player))

        boards[game_idx].play(action, currents[game_idx])
        if boards[game_idx].check_win(currents[game_idx]):
            winners[game_idx] = currents[game_idx]
            finished[game_idx] = True
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
                state, legal_actions = mcts.prepare_expand(
                    search.root,
                    boards[game_idx],
                    int(currents[game_idx]),
                )
                eval_requests.append(("root", game_idx, state, legal_actions, None))
                continue

            if mcts.search_complete(search):
                _commit_move(game_idx)
                continue

            leaf_request = mcts.simulate_until_leaf(search, boards[game_idx])
            if leaf_request is None:
                if mcts.search_complete(search):
                    _commit_move(game_idx)
                continue

            state, legal_actions = mcts.prepare_expand(
                leaf_request.node,
                leaf_request.board,
                leaf_request.to_play,
            )
            eval_requests.append(("leaf", game_idx, state, legal_actions, leaf_request))

        if not eval_requests:
            continue

        policy_batch, value_batch = mcts.evaluate_states([request[2] for request in eval_requests])
        for idx, (request_type, game_idx, _state, legal_actions, extra) in enumerate(eval_requests):
            policy_logits = policy_batch[idx]
            value = float(value_batch[idx])
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

    all_data = []
    for game_idx in range(num_games):
        winner = int(winners[game_idx])
        outcome = 1.0 if winner == Player.BLACK else -1.0
        game_data = []
        for state_np, policy_np, _player_at_pos in trajectories[game_idx]:
            game_data.append((state_np, policy_np, outcome))
        all_data.append(game_data)

    return all_data


def play_self_play_game(config, network, device):
    return generate_self_play_data(config, network, 1, device)[0]
