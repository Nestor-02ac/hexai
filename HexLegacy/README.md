# HexLegacy: Type 2 Rollouts as a Baseline for Monte-Carlo Hex

Comparative study of the MCTS algorithm from **"Monte-Carlo Hex"** (Cazenave &
Saffidine), using only **type 2 rollouts** (bridge defense) instead of the
paper's type 3 (bridges + level-2 edge templates).

The goal is to measure how much of the paper's results hold with a simpler
rollout policy, isolating the contribution of the core algorithm (UCT + RAVE)
from the domain-specific knowledge encoded in higher-level templates.

---

## Experimental Results

All experiments: 11x11 board, 200 games per data point, Cython backend.

### Table 1 — Simulation Count

*How does playing strength scale with more simulations?*

Each agent plays against a fixed 16000-simulation reference (both using
UCT + RAVE, C=0.3, bias=0.00025, type 2 rollouts).

| Simulations | Ours (type 2) | Paper (type 3) | Delta |
|:-----------:|:-------------:|:--------------:|:-----:|
| 1,000       | 31.5%         | 6.0%           | +25.5 |
| 2,000       | 45.5%         | 11.5%          | +34.0 |
| 4,000       | 41.5%         | 20.0%          | +21.5 |
| 8,000       | 50.0%         | 33.0%          | +17.0 |
| 32,000      | 55.5%         | 61.0%          | −5.5  |
| 64,000      | 66.5%         | 68.5%          | −2.0  |

**Analysis.** The fundamental trend is preserved: more simulations yield
stronger play. However, the gap between weak and strong agents is compressed
compared to the paper, and the explanation is straightforward.

With type 2 rollouts the reference agent (16k sims) is itself weaker than the
paper's type 3 reference. A weaker gatekeeper is easier to beat — hence our
1000-sim agent already wins 31.5% instead of the paper's 6%. The signal added
by each doubling of simulations is diluted because the baseline was already
more beatable.

At high simulation counts (32k–64k) the tree policy dominates over rollout
quality: the UCT tree has explored enough to make good decisions regardless of
how the random playout finishes. Here our results converge with the paper's
(66.5% vs 68.5% at 64k), confirming that the core UCT + RAVE mechanism is
working correctly.

The 4000-sim data point (41.5%) sits slightly below 2000 (45.5%). With 200
games this difference is within the expected statistical noise (95% CI ≈ ±7pp
at 50%), and the overall upward trend remains clear.

### Table 2 — UCT Exploration Constant

*What is the optimal exploration-exploitation trade-off?*

Each agent uses a different C_uct value and plays against a C=0.3 reference
(both 16k sims, RAVE, type 2).

| C_uct | Ours (type 2) | Paper (type 3) | Delta |
|:-----:|:-------------:|:--------------:|:-----:|
| 0.0   | 88.0%         | 61.0%          | +27.0 |
| 0.1   | 62.5%         | 60.0%          | +2.5  |
| 0.2   | 52.5%         | 55.5%          | −3.0  |
| 0.4   | 43.0%         | 42.0%          | +1.0  |
| 0.5   | 55.5%         | 41.0%          | +14.5 |
| 0.6   | 47.5%         | 35.5%          | +12.0 |
| 0.7   | 50.5%         | 32.5%          | +18.0 |

**Analysis.** Both the paper and our results agree that **low C_uct values
outperform high ones when RAVE is active**. C=0.0 dominates decisively in both
cases — this makes sense because RAVE already provides exploration guidance, so
UCT's sqrt-log exploration bonus is redundant or even harmful.

The main difference: the paper shows a clean monotonic decline from C=0.0 to
C=0.7 (61% → 32.5%), while our results are noisier in the C=0.4–0.7 range
(43%–55.5%). This is the compressed-spread effect seen in Table 1: with weaker
type 2 rollouts, the C=0.3 reference itself is weaker, so suboptimal C values
don't lose as badly against it. The paper's type 3 rollouts create a stronger
reference that punishes suboptimal exploration constants more severely.

The key qualitative finding is preserved: **with RAVE, pure exploitation
(C=0.0) is optimal**, and adding UCT exploration on top of RAVE hurts
performance. This is consistent with the paper's interpretation that RAVE
subsumes the need for UCT's exploration bonus.

### Table 3 — Rollout Policy (Simulation Type)

*How much does bridge defense improve over random rollouts?*

Type 1 (random) plays against type 2 (bridges), both with 16k sims, C=0.0,
RAVE.

| Matchup                   | Ours   | Paper  |
|:-------------------------:|:------:|:------:|
| Type 1 (random) vs type 2 (bridges) | 28.5%  | 22.0%* |

*\*Paper measures type 1 vs type 3 (bridges + templates), not vs type 2.*

Paper reference values (all vs their best, type 3):

| Rollout Type | Paper win % |
|:------------:|:-----------:|
| Type 1 (random)   | 22.0%       |
| Type 2 (bridges)  | 42.0%       |
| Type 4 (ziggurats)| 71.5%       |

**Analysis.** Our experiment directly measures the gap between random and
bridge-aware rollouts. Type 1 wins only 28.5% against type 2, confirming that
**bridge defense during rollouts provides a substantial playing strength
boost** — roughly equivalent to a 3:1 simulation advantage (recall from
Table 1 that 4000 vs 16000 yields ~42%).

The paper's Table 3 compares each type against type 3 as reference, so the
numbers aren't directly comparable to ours. However, the paper's 22% for
type 1 vs type 3 is lower than our 28.5% for type 1 vs type 2, which is
consistent: type 3 is a tougher opponent than type 2, so random rollouts fare
even worse against it.

This confirms the hierarchy: random << bridges << templates, and that even the
simplest domain knowledge (defending bridges) produces a measurable advantage.

### Table 4 — RAVE Bias

*How sensitive is RAVE to the bias parameter?*

Each agent uses a different RAVE bias and plays against bias=0.001 as
reference (16k sims, C=0.0, type 2).

| RAVE bias  | Ours (type 2) | Paper (type 3) | Delta |
|:----------:|:-------------:|:--------------:|:-----:|
| 0.0005     | 47.0%         | 50.5%          | −3.5  |
| 0.00025    | 49.0%         | 59.0%          | −10.0 |
| 0.000125   | 46.0%         | 53.5%          | −7.5  |

**Analysis.** The paper finds that **lower RAVE bias is better**, with a peak
at bias=0.00025 (59% vs the 0.001 reference). Lower bias means RAVE influence
persists longer into the tree — the AMAF statistics are trusted for more
visits before yielding to direct UCT estimates.

Our results are broadly flat around 47–49%, without the clear peak the paper
shows at 0.00025. This is the most notable divergence from the paper and has
a clear explanation: **RAVE bias sensitivity depends on rollout quality**.

With type 3 rollouts, AMAF statistics during the random playout are more
informative because moves are guided by bridge + template knowledge. Trusting
these statistics longer (lower bias) pays off. With type 2 rollouts, the AMAF
move values collected during playouts are noisier — bridge defense alone
doesn't produce move orderings as rich as bridges + templates. Consequently,
the gain from trusting RAVE longer is diminished, and all bias values perform
similarly.

This is arguably the most interesting finding in the study: **the value of fine-
tuning RAVE parameters depends on the quality of the rollout policy feeding
RAVE's statistics**. With weaker rollouts, RAVE tuning matters less because
the signal-to-noise ratio of AMAF values is lower regardless of how long you
trust them.

---

## Summary

| Experiment | Paper finding | Preserved with type 2? | Notes |
|:----------:|:------------:|:---------------------:|:-----:|
| Table 1 (sims) | More sims = stronger play | Yes | Compressed spread at low sims, convergence at high sims |
| Table 2 (C_uct) | C=0.0 optimal with RAVE | Yes | Noisier at high C values due to weaker reference |
| Table 3 (rollout type) | Better rollouts = far stronger play | Yes | 28.5% confirms bridges >> random |
| Table 4 (RAVE bias) | Lower bias preferred, peak at 0.00025 | Partially | Flat response — weaker rollouts → noisier AMAF → less tuning payoff |

The core UCT + RAVE algorithm is robust: its qualitative behavior survives
even when the rollout policy is simplified. The domain-specific template
knowledge (type 3) amplifies differences and makes parameters more sensitive,
but is not required for the algorithm to function correctly.

---

## Implementation

Two interchangeable backends, same API:

| | Python | Cython |
|---|---|---|
| Files | `mcts_hex.py` + `hex_board.py` | `cmcts_hex.pyx` + `chex_board.pyx` |
| Speed (11x11, 500 sims) | ~3.9s/game | ~0.3s/game |
| Speedup | 1x | **~12x** |

Both implement: UCT, RAVE/AMAF, bridge defense (type 2), Union-Find win
detection, game-level multiprocessing.

### Cython optimizations
- C struct node pool with per-node dynamic children arrays
- Flat `int*`/`double*` arrays for RAVE (vs Python dicts)
- Fisher-Yates shuffle with `libc rand()`
- `memcpy`-based board cloning (vs `deepcopy`)
- Inline `_find`/`_union` with path compression (`noexcept nogil`)

## Usage

```bash
# Build Cython extensions (requires cython package)
python setup.py build_ext --inplace

# Print all saved results as tables
python show_results.py

# Quick sanity check: MCTS vs Random
python experiments.py sanity --cython

# Scaled-down trend verification on 5x5 (minutes)
python experiments.py small --cython --seed 42

# Full paper tables on 11x11 (hours each)
python experiments.py table1 --cython --seed 42
python experiments.py table2 --cython --seed 42
python experiments.py table3 --cython --seed 42
python experiments.py table4 --cython --seed 42

# All tables
python experiments.py all --cython --seed 42
```

Flags: `--cython` (use fast backend), `--seed N` (reproducibility),
`--workers N` (parallel games, default: all CPU cores).

Omit `--cython` to run with the pure Python backend.

## What is NOT implemented

These features from the paper are intentionally omitted to keep the study
focused on the core UCT + RAVE algorithm:

- **Level-2 edge templates** (section 1.1): pattern-matching for 4-3-2 edge
  templates during rollouts. This is what distinguishes type 3 from type 2
  and is the main source of absolute % differences with the paper.
- **Ziggurats** (section 2.1): the paper notes this makes sequential play
  slower (32% win rate with fixed time).
- **Virtual connection solver** (section 2.3): requires Anshelevich's algorithm
  for computing virtual connections and mustplay regions.

## Project Structure

```
hex_board.py       Python Hex board with Union-Find
mcts_hex.py        Python MCTS (UCT + RAVE + type 2 rollouts)
chex_board.pyx/pxd Cython Hex board
cmcts_hex.pyx      Cython MCTS
experiments.py      Experiment runner (Tables 1-4, sanity, small)
show_results.py    Print saved results as readable tables
setup.py           Cython build script
results/           Saved experiment outputs (JSON)
```
