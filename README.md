# MCTS

Monte-Carlo Tree Search implementations for board games.

## Projects

### [HexLegacy/](HexLegacy/)

Comparative study of the MCTS algorithm from "Monte-Carlo Hex" (Cazenave &
Saffidine), measuring how well the paper's results hold using only type 2
rollouts (bridge defense) vs the paper's type 3 (bridges + level-2 templates).

Implements UCT + RAVE with both Python and Cython (~12x faster) backends.
See [HexLegacy/README.md](HexLegacy/README.md) for findings and usage.
