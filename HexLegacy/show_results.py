"""Print saved experiment results as readable tables."""

import json
import glob
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_latest(prefix):
    """Find the most recent result file matching a prefix."""
    pattern = os.path.join(RESULTS_DIR, f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def fmt_pct(val):
    return f"{val:.1f}%"


def print_table(headers, rows, col_widths=None):
    """Print a simple aligned table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for row in rows:
                w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    header_line = "".join(h.center(w) for h, w in zip(headers, col_widths))
    sep = "".join("-" * w for w in col_widths)
    print(f"  {header_line}")
    print(f"  {sep}")
    for row in rows:
        line = "".join(str(v).center(w) for v, w in zip(row, col_widths))
        print(f"  {line}")


def show_table1(data):
    print("Table 1: Simulation count (vs 16k reference)")
    print(f"  11x11, 200 games, UCT+RAVE, C=0.3, type 2 rollouts\n")

    paper = data["data"]["paper_results"]
    ours = data["data"]["our_results"]

    headers = ["Sims", "Ours", "Paper", "Delta"]
    rows = []
    for sims in ["1000", "2000", "4000", "8000", "32000", "64000"]:
        if sims not in ours:
            continue
        o = ours[sims]["win_pct"]
        p = paper[sims]
        delta = o - p
        sign = "+" if delta >= 0 else ""
        rows.append((f"{int(sims):,}", fmt_pct(o), fmt_pct(p), f"{sign}{delta:.1f}"))
    print_table(headers, rows)
    print()


def show_table2(data):
    print("Table 2: UCT exploration constant (vs C=0.3 reference)")
    print(f"  11x11, 200 games, 16k sims, RAVE, type 2 rollouts\n")

    paper = data["data"]["paper_results"]
    ours = data["data"]["our_results"]

    headers = ["C_uct", "Ours", "Paper", "Delta"]
    rows = []
    for c in ["0.0", "0.1", "0.2", "0.4", "0.5", "0.6", "0.7"]:
        if c not in ours:
            continue
        o = ours[c]["win_pct"]
        p = paper[c]
        delta = o - p
        sign = "+" if delta >= 0 else ""
        rows.append((c, fmt_pct(o), fmt_pct(p), f"{sign}{delta:.1f}"))
    print_table(headers, rows)
    print()


def show_table3(data):
    print("Table 3: Rollout policy (random vs bridges)")
    print(f"  11x11, 200 games, 16k sims, C=0.0, RAVE\n")

    ours = data["data"]["our_results"]
    paper = data["data"]["paper_results"]

    o = ours["random_vs_bridges"]["win_pct"]

    headers = ["Matchup", "Ours (vs type 2)", "Paper (vs type 3)"]
    rows = [
        ("Random", fmt_pct(o), fmt_pct(paper["type1_random"])),
    ]
    print_table(headers, rows)

    print(f"\n  Paper reference (all vs type 3):")
    for name, key in [("Random", "type1_random"), ("Bridges", "type2_bridges"), ("Ziggurats", "type4_ziggurats")]:
        print(f"    {name:>10}: {fmt_pct(paper[key])}")
    print()


def show_table4(data):
    print("Table 4: RAVE bias (vs bias=0.001 reference)")
    print(f"  11x11, 200 games, 16k sims, C=0.0, type 2 rollouts\n")

    paper = data["data"]["paper_results"]
    ours = data["data"]["our_results"]

    headers = ["Bias", "Ours", "Paper", "Delta"]
    rows = []
    for b in ["0.0005", "0.00025", "0.000125"]:
        if b not in ours:
            continue
        o = ours[b]["win_pct"]
        p = paper[b]
        delta = o - p
        sign = "+" if delta >= 0 else ""
        rows.append((b, fmt_pct(o), fmt_pct(p), f"{sign}{delta:.1f}"))
    print_table(headers, rows)
    print()


def main():
    tables = {
        "table1_simulations": show_table1,
        "table2_uct_constant": show_table2,
        "table3_templates": show_table3,
        "table4_rave_bias": show_table4,
    }

    # If args given, show only those tables
    which = sys.argv[1:] if len(sys.argv) > 1 else list(tables.keys())

    found_any = False
    for name in which:
        key = name if name in tables else f"table{name}" if f"table{name}" in tables else None
        # try matching by number: "1" -> "table1_simulations"
        if key is None:
            for k in tables:
                if name in k:
                    key = k
                    break
        if key is None:
            print(f"Unknown table: {name}")
            continue

        data = load_latest(key)
        if data is None:
            print(f"No results found for {key} in {RESULTS_DIR}/")
            continue

        found_any = True
        print()
        tables[key](data)

    if not found_any:
        print(f"No result files found in {RESULTS_DIR}/")
        print("Run experiments first: python experiments.py table1 --cython")


if __name__ == "__main__":
    main()
