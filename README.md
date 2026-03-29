# MCTS for Hex

Exploring Monte Carlo Tree Search approaches for the game of Hex, from
classical UCT + RAVE to learned evaluation (Gumbel AlphaZero). Each approach
lives in its own directory with shared benchmarking so they can be compared
head-to-head.

![HexClassic vs HexGumbel dashboard](visualization/hex_dashboard_7x7.gif)

---

## 1. Classical MCTS — `HexClassic/`

Comparative study of the algorithm from **"Monte-Carlo Hex"** (Cazenave &
Saffidine). We implement UCT + RAVE with **type 2 rollouts** (bridge defense
only) and measure how well the paper's results — originally obtained with
type 3 rollouts (bridges + level-2 edge templates) — hold under a simpler
rollout policy.

The point is to isolate the contribution of the core search algorithm from the
domain-specific knowledge baked into higher-level templates.

### Results

All experiments: 11×11 board, 200 games per data point, Cython backend.

#### Table 1 — Simulation count

*How does playing strength scale with more simulations?*

Each agent plays against a 16k-sim reference (UCT + RAVE, C=0.3, bias=0.00025,
type 2).

| Simulations | Ours (type 2) | Paper (type 3) | Delta |
|:-----------:|:-------------:|:--------------:|:-----:|
| 1,000       | 31.5%         | 6.0%           | +25.5 |
| 2,000       | 45.5%         | 11.5%          | +34.0 |
| 4,000       | 41.5%         | 20.0%          | +21.5 |
| 8,000       | 50.0%         | 33.0%          | +17.0 |
| 32,000      | 55.5%         | 61.0%          | −5.5  |
| 64,000      | 66.5%         | 68.5%          | −2.0  |

The trend is preserved: more simulations = stronger play. The spread is
compressed at low sim counts because our type 2 reference is weaker (easier to
beat). At 32k–64k the tree policy dominates rollout quality and our results
converge with the paper's (66.5% vs 68.5%).

#### Table 2 — UCT exploration constant

*What is the optimal exploration-exploitation trade-off?*

Each agent varies C_uct against a C=0.3 reference (16k sims, RAVE, type 2).

| C_uct | Ours (type 2) | Paper (type 3) | Delta |
|:-----:|:-------------:|:--------------:|:-----:|
| 0.0   | 88.0%         | 61.0%          | +27.0 |
| 0.1   | 62.5%         | 60.0%          | +2.5  |
| 0.2   | 52.5%         | 55.5%          | −3.0  |
| 0.4   | 43.0%         | 42.0%          | +1.0  |
| 0.5   | 55.5%         | 41.0%          | +14.5 |
| 0.6   | 47.5%         | 35.5%          | +12.0 |
| 0.7   | 50.5%         | 32.5%          | +18.0 |

Both agree: **C=0.0 is optimal when RAVE is active**. RAVE already handles
exploration, so UCT's bonus becomes redundant or harmful. The paper shows a
cleaner monotonic decline because its stronger type 3 reference punishes
suboptimal C values more harshly.

#### Table 3 — Rollout policy

*How much does bridge defense improve over random?*

Type 1 (random) vs type 2 (bridges), 16k sims, C=0.0, RAVE.

| Matchup                              | Ours   | Paper  |
|:-------------------------------------:|:------:|:------:|
| Type 1 (random) vs type 2 (bridges)  | 28.5%  | 22.0%* |

*\*Paper measures type 1 vs type 3 (bridges + templates), not vs type 2.*

Random wins only 28.5% against bridges — roughly equivalent to a 3:1
simulation disadvantage. The paper's 22% is even lower because their reference
(type 3) is tougher. This confirms the hierarchy: random << bridges <<
templates, and that even the simplest domain knowledge makes a real difference.

#### Table 4 — RAVE bias

*How sensitive is RAVE to the bias parameter?*

Each agent varies RAVE bias against bias=0.001 (16k sims, C=0.0, type 2).

| RAVE bias  | Ours (type 2) | Paper (type 3) | Delta |
|:----------:|:-------------:|:--------------:|:-----:|
| 0.0005     | 47.0%         | 50.5%          | −3.5  |
| 0.00025    | 49.0%         | 59.0%          | −10.0 |
| 0.000125   | 46.0%         | 53.5%          | −7.5  |

The paper finds a clear peak at bias=0.00025. Our results are flat (~47–49%).
This is the most interesting divergence: **RAVE tuning sensitivity depends on
rollout quality**. With type 3, AMAF statistics are more informative so
trusting them longer (lower bias) pays off. With type 2, the AMAF signal is
noisier, so the payoff from precise tuning disappears.

#### Summary

| Experiment | Paper finding | Preserved? | Notes |
|:----------:|:------------:|:----------:|:-----:|
| Sim count | More sims = stronger | Yes | Compressed at low sims, converges at high sims |
| C_uct | C=0.0 optimal with RAVE | Yes | Noisier at high C (weaker reference) |
| Rollout type | Better rollouts = stronger | Yes | 28.5% confirms bridges >> random |
| RAVE bias | Lower bias preferred | Partially | Flat — noisier AMAF reduces tuning payoff |

The core UCT + RAVE algorithm is robust: its qualitative behavior holds even
with a simpler rollout policy. Type 3 templates amplify differences and sharpen
parameter sensitivity, but aren't needed for the algorithm itself to work.

### Implementation

Two interchangeable backends:

| | Python | Cython |
|---|---|---|
| Files | `mcts_hex.py` + `hex_board.py` | `cmcts_hex.pyx` + `chex_board.pyx` |
| Speed (11×11, 500 sims) | ~3.9s/game | ~0.3s/game |
| Speedup | 1× | **~12×** |

Cython gains come from: C struct node pool, flat RAVE arrays, Fisher-Yates
shuffle via `libc rand()`, `memcpy` board cloning, inline Union-Find with path
compression.

### Not implemented

Intentionally omitted to keep the study focused on the core algorithm:

- **Level-2 edge templates** — the type 3 vs type 2 distinction. Main source
  of absolute % differences with the paper.
- **Ziggurats** — paper notes this hurts sequential play speed.
- **Virtual connection solver** — requires Anshelevich's VC algorithm.

### Usage

```bash
cd HexClassic/

# build cython (needs cython package)
python setup.py build_ext --inplace

# print saved results
python show_results.py

# quick test: MCTS vs random
python experiments.py sanity --cython

# small-scale trend check (5×5, minutes)
python experiments.py small --cython --seed 42

# full tables on 11×11 (hours each)
python experiments.py table1 --cython --seed 42
python experiments.py table2 --cython --seed 42
python experiments.py table3 --cython --seed 42
python experiments.py table4 --cython --seed 42
python experiments.py all --cython --seed 42
```

Flags: `--cython` (fast backend), `--seed N` (reproducibility), `--workers N`
(parallel games, default: all cores). Omit `--cython` for pure Python.

### Files

```
HexClassic/
  hex_board.py       Python board with Union-Find
  mcts_hex.py        Python MCTS (UCT + RAVE + type 2)
  chex_board.pyx/pxd Cython board
  cmcts_hex.pyx      Cython MCTS
  experiments.py     Experiment runner (tables 1-4, sanity, small)
  show_results.py    Print saved results as tables
  setup.py           Cython build script
  results/           Experiment outputs (JSON)
```

---

## 2. Gumbel AlphaZero — `HexGumbel/`

Neural-network-guided MCTS trained via self-play. Instead of rollouts and RAVE,
a ResNet provides move probabilities (policy) and position evaluation (value)
directly. Uses the **Gumbel AlphaZero** search policy at the root — Gumbel noise + sequential halving for action selection — with
standard PUCT at interior nodes. However is not entirely the zero version since given our compute budget we decided to give a stronger prior distribution via a supervised learning phase. To also accomodate further we decided to use a smaller board size (7x7).

### How it works

1. **Board encoding** — 4 binary planes: current player's stones, opponent's
   stones, Black-to-move indicator, White-to-move indicator.

2. **Network** — ResNet (20 blocks, 192 channels for 7×7). Splits into a
   policy head (move logits over all board cells) and a value head (tanh scalar
   in [−1, 1]).

3. **Gumbel MCTS at the root** —
   - Sample Gumbel(0,1) noise for each legal move
   - Score moves by `g(a) + log π(a)` and keep the top-*m* candidates
   - Sequential halving: run simulations in phases, eliminate the bottom half
     each phase
   - Final move selection uses the completed Q-values `σ(q̄)`

4. **PUCT at interior nodes** — Standard AlphaZero-style selection below root.

5. **Improved policy** — Training target is `softmax(log π + σ(q̄))`, trained
   with KL divergence. This is superior to raw visit-count targets because it
   incorporates value information.

### Training pipeline

The full pipeline has three stages:

#### Stage 1: Generate expert data with classical MCTS

```bash
cd HexClassic/
python generate_expert_data.py --games 1000 --sims 16000 --board-size 7 \
  --output ../HexGumbel/data/expert_data_16k.jsonl
```

Uses the Cython MCTS (UCT + RAVE + bridge rollouts) to produce high-quality
move labels. Each game generates ~30–40 positions as (board, expert_move,
player) tuples.

#### Stage 2: Pretrain the neural network (supervised)

```bash
cd HexGumbel/
python pretrain_supervised.py --data data/expert_data_16k.jsonl \
  --channels 192 --res-blocks 20 --board-size 7 \
  --epochs 5 --batch-size 128 --lr 0.001 \
  --output checkpoints/pretrained_model_16k.pt
```

Trains the policy head via cross-entropy on the expert moves. This gives the
network a reasonable starting point so that self-play games aren't random at the
beginning.

#### Stage 3: Self-play reinforcement learning (Gumbel Zero)

```bash
python train.py \
  --resume checkpoints/pretrained_model_16k.pt \
  --board-size 7 --channels 192 --res-blocks 20 \
  --simulations 16 --gumbel-sample-size 8 \
  --gumbel-sigma-visit-c 16 --gumbel-sigma-scale-c 1.0 \
  --pb-c-base 19652 --pb-c-init 1.25 \
  --lr 0.001 --weight-decay 0.0001 --batch-size 2048 --train-steps 8 \
  --value-loss-weight 1.0 --replay-capacity 500000 \
  --iterations 70 --games-per-iter 512 \
  --eval-games 40 --eval-interval 5 --eval-mcts-simulations 16000 \
  --seed 42 --no-lr-scheduler
```

Each iteration: 512 self-play games → 8 gradient steps (~1 epoch of new data)
→ evaluation. The low step count prevents overfitting to the replay buffer.

#### Standalone evaluation

```bash
python eval_checkpoint.py \
  --checkpoint checkpoints/<run_id>/iter_0070.pt \
  --board-size 7 --channels 192 --res-blocks 20 \
  --simulations 16 --gumbel-sample-size 8 \
  --gumbel-sigma-visit-c 16 --gumbel-sigma-scale-c 1.0 \
  --pb-c-base 19652 --pb-c-init 1.25 \
  --games 40 --mcts-sims 16000 --seed 42
```

### Results

7×7 board. Gumbel agent uses **only 16 simulations** per move. Evaluated
against classical MCTS with **16,000 simulations** (UCT + RAVE + bridges) — a
**1000× simulation budget disadvantage**.

#### Training progression (vs 16K-sim classical MCTS, 40 games each)

| Iteration | vs Random | vs MCTS 16K | Buffer size |
|:---------:|:---------:|:-----------:|:-----------:|
| 5         | 100%      | 10.0%       |  69,220     |
| 10        | 100%      | 32.5%       | 132,931     |
| 20        | 100%      | 45.0%       | 255,492     |
| 30        | 100%      | 55.0%       | 121,818     |
| 40        | 100%      | 62.5%       |  63,061     |
| 50        | 100%      | 55.0%       |  61,491     |
| 60        | 100%      | 50.0%       |  62,208     |
| **70**    | **100%**  | **70.0%**   | 188,366     |

The agent reaches a **65–70% win rate** against an opponent running 1000×
more simulations, demonstrating that a learned evaluation network with
Gumbel-guided search can dramatically outperform classical rollout-based
MCTS on Hex.

#### Key design decisions

- **16 simulations** with 8 Gumbel candidates — very low sim budget forces the
  network to produce strong policies directly rather than relying on deep
  search.
- **σ_visit_c = 16** — lower than the default 50; with few simulations, a
  smaller sigma prevents over-weighting of noisy Q-estimates.
- **~1 epoch per iteration** (`train_steps ≈ buffer / batch_size`) — prevents
  overfitting to the small replay buffer, which was a key failure mode during
  development.
- **Supervised pretraining** — bootstraps the policy from classical MCTS expert
  data, avoiding the cold-start problem where random self-play produces
  uninformative training signals.

### Files

```
HexGumbel/
  config.py              All hyperparameters (dataclass)
  neural_net.py          ResNet with policy + value heads
  mcts.py                Gumbel MCTS search (Python)
  cgumbel_mcts.pyx       Gumbel MCTS search (Cython, ~12× faster)
  self_play.py           Game generation + replay buffer
  trainer.py             Training loop (self-play → train → eval)
  evaluate.py            Benchmarks vs random and classical MCTS
  train.py               CLI entry point for training
  eval_checkpoint.py     CLI entry point for standalone evaluation
  pretrain_supervised.py Supervised pretraining on expert data
  hex_board.py           Python board (shared with HexClassic)
  chex_board.pyx/pxd     Cython board (shared with HexClassic)
  setup.py               Cython build script
  data/                  Expert training data (JSONL)
  checkpoints/           Saved model checkpoints + pretrained models
```

---

## 3. Visualization — `visualization/`

Dashboard GIF generator comparing HexClassic and HexGumbel side-by-side.

```bash
# Generate the comparison dashboard GIF
python visualization/generate_dashboard_gif.py

# Legacy layout variant
python visualization/generate_dashboard_gif_v1.py
```

```
visualization/
  generate_dashboard_gif.py     Dark-theme 2×2 dashboard GIF
  generate_dashboard_gif_v1.py  Light-theme legacy layout
  hex_dashboard_5x5.gif         Output GIF (shown in README)
  hex_dashboard_5x5_v1.gif      Output GIF (legacy layout)
```

---

## Setup

```bash
# Create conda environment
conda create -n hex python=3.11
conda activate hex
pip install torch numpy cython tqdm

# Build Cython extensions (both modules)
cd HexClassic && python setup.py build_ext --inplace && cd ..
cd HexGumbel && python setup.py build_ext --inplace && cd ..
```

## References

- **Monte-Carlo Hex** — T. Cazenave, A. Saffidine (2009). UCT + RAVE with
  typed rollouts for Hex.
- **Policy improvement by planning with Gumbel** — I. Danihelka et al. (2022).
  Gumbel sampling + sequential halving for root action selection in AlphaZero.
- **Mastering the game of Go without human knowledge** — D. Silver et al.
  (2017). AlphaZero self-play framework.
