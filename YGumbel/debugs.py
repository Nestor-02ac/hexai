from y_board import YBoard, Player
from neural_net import YNet
from config import GumbelZeroConfig
from mcts import create_gumbel_mcts
import torch

config = GumbelZeroConfig(board_size=5)
device = torch.device("cpu")

net = YNet(config.board_size).to(device)
mcts = create_gumbel_mcts(config, net, device)

board = YBoard(5)
current = Player.BLACK

for move_idx in range(30):  # safety limit
    print(f"\nMove {move_idx+1} — Player {current}")

    action, policy = mcts.run(board, int(current))

    r, c = board.idx_to_rc(action)
    print(f"Chosen move: ({r},{c})")

    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Max prob: {policy.max():.4f}")

    # sanity check
    if board.board[action] != 0:
        print("ILLEGAL MOVE")
        break

    board.play(action, current)
    board.display()

    if board.check_win(current):
        print(f"WINNER: {current}")
        break

    current = Player.WHITE if current == Player.BLACK else Player.BLACK