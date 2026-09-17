"""Adopt the 16-pipeline IV-2a core run, but only if it reproduces the old one.

Referee item 9: Table 3 had 15 IV-2a rows while its caption and the Spearman
correlation spoke of 16, because quantum/QRE-RBF-SVM joined the core suite
after IV-2a had been run. scripts/rerun_bci2a_core16.sh reruns IV-2a with the
current suite under a new tag. This script decides whether that run may
extend the published file.

The rule is strict. The 15 pipelines the two runs share must give the same
accuracy in every subject and fold and select the same hyperparameters.
Anything else means the code or the environment has drifted since the
published run, and that is a problem to understand, not a result to adopt.

AUC gets one allowance, and it is reported rather than hidden. A fold's AUC
moves in steps of 1/(n_pos * n_neg), and two test trials whose decision values
are equal to machine precision can swap order between runs. The first check of
this rerun found exactly that: one CNOT-kernel fold of 30, AUC 0.5636 against
0.5624, a difference of 1/841, with identical accuracy. So a fold may differ by
at most one such step, and each pipeline's mean AUC must be unchanged at the
three decimals the tables print.

Adoption never rewrites a published row. The adopted file is the published
rows for the shared pipelines plus the new pipeline's rows from the rerun, so
every number already in the paper is carried over exactly.

    python scripts/adopt_bci2a_core16.py              # compare, change nothing
    python scripts/adopt_bci2a_core16.py --partial    # compare the checkpoint so far
    python scripts/adopt_bci2a_core16.py --adopt      # compare, then extend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NEW_TAG = "bci2a_core16"
OLD_TAG = "bci2a_motor8_q4"
KEY = ["subject", "pipeline", "fold"]
TOL = 1e-9
# One swapped pair in a test fold of about 58 balanced trials (29 x 29 pairs)
# moves AUC by 1/841 = 0.0012. This admits one swap and nothing larger.
AUC_STEP = 2e-3


def compare(new: pd.DataFrame, old: pd.DataFrame) -> bool:
    shared = sorted(set(new.pipeline) & set(old.pipeline))
    added = sorted(set(new.pipeline) - set(old.pipeline))
    dropped = sorted(set(old.pipeline) - set(new.pipeline))
    subjects = sorted(set(new.subject) & set(old.subject))
    print(f"subjects compared : {subjects}")
    print(f"shared pipelines  : {len(shared)}")
    print(f"new in this run   : {added or 'none'}")
    if dropped:
        print(f"MISSING from the new run: {dropped}")

    a = new[new.pipeline.isin(shared) & new.subject.isin(subjects)]
    b = old[old.pipeline.isin(shared) & old.subject.isin(subjects)]
    m = a.merge(b, on=KEY, suffixes=("_new", "_old"), how="outer",
                indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"]

    ok = not dropped and unmatched == 0
    if unmatched:
        print(f"MISMATCHED rows (present in one run only): {unmatched}")
    header = f"{'pipeline':40s} {'max |d acc|':>12s} {'max |d auc|':>12s}"
    print()
    print(f"{header} {'mean AUC old/new':>17s} params")
    for p, g in m.groupby("pipeline"):
        da = (g.accuracy_new - g.accuracy_old).abs().max()
        d_auc = (g.auc_new - g.auc_old).abs()
        du = d_auc.max()
        mean_new = g.groupby("subject").auc_new.mean().mean()
        mean_old = g.groupby("subject").auc_old.mean().mean()
        same_params = bool((g.best_params_new.astype(str)
                            == g.best_params_old.astype(str)).all())
        good = (da <= TOL and du <= AUC_STEP and same_params
                and f"{mean_new:.3f}" == f"{mean_old:.3f}")
        ok &= good
        if not good:
            note = "   <-- DIFFERS"
        elif du > TOL:
            note = (f"   AUC differs in {int((d_auc > TOL).sum())} fold(s), "
                    f"at most one tied pair each")
        else:
            note = ""
        print(f"{p:40s} {da:12.2e} {du:12.2e} {mean_old:8.3f}/{mean_new:.3f} "
              f"{'same' if same_params else 'DIFFER'}{note}")
    print(f"\n{len(m)} fold rows compared; the shared pipelines "
          f"{'reproduce' if ok else 'do NOT reproduce'} the published run")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--partial", action="store_true",
                    help="compare the .partial.csv checkpoint of a running job")
    ap.add_argument("--adopt", action="store_true",
                    help="extend the published IV-2a core files if reproduced")
    args = ap.parse_args(argv)

    new_f = RES / (f"raw_folds_{NEW_TAG}.partial.csv" if args.partial
                   else f"raw_folds_{NEW_TAG}.csv")
    old_f = RES / f"raw_folds_{OLD_TAG}.csv"
    if not new_f.exists():
        print(f"{new_f.name} not found; is the rerun finished?")
        return 1
    new, old = pd.read_csv(new_f), pd.read_csv(old_f)
    ok = compare(new, old)
    if not args.adopt:
        return 0 if ok else 1
    if args.partial:
        print("refusing to adopt a checkpoint; wait for the run to finish")
        return 1
    if not ok:
        print("refusing to adopt: the shared pipelines do not reproduce")
        return 1
    if set(new.subject) != set(old.subject):
        print(f"refusing to adopt: subjects differ "
              f"({sorted(set(new.subject))} vs {sorted(set(old.subject))})")
        return 1

    sys.path.insert(0, str(ROOT / "src"))
    from qeeg.benchmark import paired_tests, summarise

    old_meta = json.loads((RES / f"meta_{OLD_TAG}.json").read_text())
    new_meta = json.loads((RES / f"meta_{NEW_TAG}.json").read_text())
    reference = old_meta.get("reference", "classical/TS+LR")

    added = sorted(set(new.pipeline) - set(old.pipeline))
    extra = new[new.pipeline.isin(added)]
    adopted = pd.concat([old, extra], ignore_index=True)
    # Append the new rows to the published bytes rather than rewriting the
    # file through pandas: a round trip reprints some float columns in their
    # last digit (fit times moved by 4e-15), which leaves every published row
    # numerically identical but shows as ~2000 changed lines in git.
    body = old_f.read_bytes()
    if not body.endswith(b"\n"):
        body += b"\n"
    old_f.write_bytes(body + extra.to_csv(header=False, index=False,
                                          lineterminator="\n").encode("utf-8"))
    summarise(adopted).to_csv(RES / f"summary_{OLD_TAG}.csv", index=False)
    paired_tests(adopted, reference).to_csv(
        RES / f"tests_vs_{reference.replace('/', '-')}_{OLD_TAG}.csv", index=False)
    meta = {**old_meta, **new_meta,
            "adopted_from": f"raw_folds_{NEW_TAG}.csv",
            "note": (f"published rows kept for the shared pipelines; rows for "
                     f"{', '.join(added)} added from a rerun with the "
                     f"16-pipeline core suite, whose shared pipelines reproduced "
                     f"the published accuracies exactly "
                     f"(scripts/adopt_bci2a_core16.py)")}
    (RES / f"meta_{OLD_TAG}.json").write_text(json.dumps(meta, indent=2))
    for f in (RES / f"raw_folds_{NEW_TAG}.csv", RES / f"meta_{NEW_TAG}.json"):
        f.unlink(missing_ok=True)
    print(f"adopted: {old_f.name} now holds {adopted.pipeline.nunique()} "
          f"pipelines; rerun python paper/make_tables.py and the figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
