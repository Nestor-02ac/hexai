# YClassic

Classical MCTS for the game of Y with matching Python and Cython backends.

The main documentation for this module now lives in the top-level
[README.md](../README.md) and in
[report/report.tex](../report/report.tex). This local file is
kept intentionally short to avoid repeating the same project-level material.

## Contents

- `y_board.py`: Python Y board with Union-Find and 3-bit side masks
- `mcts_y.py`: Python UCT + RAVE search with connectivity rollouts
- `cy_board.pyx` / `cmcts_y.pyx`: Cython fast backend
- `experiments_y.py`: Sanity checks and parameter sweeps
- `play_y.py`: Interactive play against the Y agent
- `test_y_logic.py`: Logic and Python/Cython consistency tests

## Quick Start

```bash
cd YClassic/
python3 setup.py build_ext --inplace
python3 experiments_y.py sanity --cython
python3 experiments_y.py all --cython --seed 42 --workers 20
python3 play_y.py --cython --size 7 --sims 3000
```

Full experiment runs save JSON outputs automatically in `YClassic/results/`.
