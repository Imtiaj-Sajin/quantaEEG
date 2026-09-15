"""Build reference_gram_sweep.csv: Gram variance versus register size, both frames.

`python -m qeeg.reference --gram --channels C` writes one per-subject file per
channel set. This module averages each over subjects and stacks them by qubit
count, which is the table and figure the paper shows. It replaces a one-off
script that was never committed, so the sweep can now be rebuilt from the
per-subject files by anyone.

    PYTHONPATH=src python -m qeeg.gram_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHANNEL_FILES = {"motor8": 8, "motor16": 16, "motor32": 32, "all64": 64}


def build(results: Path) -> pd.DataFrame:
    rows = []
    for tag, n_ch in CHANNEL_FILES.items():
        f = results / f"reference_gram_{tag}.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f)
        stats = g.groupby(["kernel", "frame"])[["var", "mean"]].mean().unstack("frame")
        for kernel in stats.index:
            sv, rv = stats.loc[kernel, ("var", "sensor")], stats.loc[kernel, ("var", "reference")]
            rows.append({
                "qubits": int(np.log2(n_ch)), "channels": tag, "kernel": kernel,
                "sensor_var": sv, "ref_var": rv, "gain": rv / sv,
                "sensor_mean": stats.loc[kernel, ("mean", "sensor")],
                "ref_mean": stats.loc[kernel, ("mean", "reference")],
                "n": int(g.subject.nunique()),
            })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/reference_gram_sweep.csv")
    args = ap.parse_args(argv)
    df = build(Path(args.results))
    if df.empty:
        print("no reference_gram_*.csv files found")
        return 1
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
