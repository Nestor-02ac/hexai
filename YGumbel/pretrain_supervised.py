"""
Supervised pretraining for Y neural net using expert data.

Usage:
  python pretrain_supervised_y.py
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from neural_net import YNet
import numpy as np


def load_expert_dataset(path):
    boards, moves, players = [], [], []
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Expert data not found at {p.resolve()}.")

    with p.open("r") as f:
        for line in f:
            d = json.loads(line)
            boards.append(d["board"])   # flat list
            moves.append(d["move"])
            players.append(d["player"])

    return boards, moves, players


def make_dataloader(boards, moves, players, batch_size):
    X = torch.tensor(boards, dtype=torch.float32)
    y = torch.tensor(moves, dtype=torch.long)
    p = torch.tensor(players, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y, p)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    parser = argparse.ArgumentParser(description='Supervised pretraining for YNet')
    parser.add_argument('--data', type=str, default='expert_data_16k.jsonl')
    parser.add_argument('--output', type=str, default='pretrained_y_model.pt')
    parser.add_argument('--board-size', type=int, default=5)
    parser.add_argument('--channels', type=int, default=192)
    parser.add_argument('--res-blocks', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    board_size = args.board_size
    n_cells = board_size * (board_size + 1) // 2

    boards, moves, players = load_expert_dataset(args.data)
    print(f"  Loaded {len(boards)} positions from {args.data}")

    dataloader = make_dataloader(boards, moves, players, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YNet(board_size, num_channels=args.channels,
                   num_res_blocks=args.res_blocks).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    
    def encode_board_from_array(arr, current_player: int):

        state = np.zeros((4, board_size, board_size), dtype=np.float32)
        opponent = 3 - int(current_player)

        idx = 0
        for r in range(board_size):
            for c in range(r + 1):
                cell = int(arr[idx])

                if cell == int(current_player):
                    state[0, r, c] = 1.0
                elif cell == opponent:
                    state[1, r, c] = 1.0

                idx += 1

        if int(current_player) == 1:
            state[2, :, :] = 1.0
        else:
            state[3, :, :] = 1.0

        return torch.from_numpy(state)

    num_epochs = 1 if args.test else args.epochs

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (X, y, p) in enumerate(dataloader):

            X_enc = torch.stack([
                encode_board_from_array(x.numpy(), int(player))
                for x, player in zip(X, p)
            ])

            X_enc = X_enc.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            policy_logits, _ = model(X_enc)

            loss = criterion(policy_logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

            if args.test:
                print(f"Test batch: X_enc {X_enc.shape}, logits {policy_logits.shape}, loss {loss.item():.4f}")
                break

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"  Pretrained model saved to {args.output}")


if __name__ == "__main__":
    main()