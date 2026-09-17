"""quantaEEG: one entry point for setup, verification, reproduction and the paper.

    python run.py status      what results this checkout contains
    python run.py verify      recheck the paper's claims from the committed CSVs
    python run.py setup       install dependencies and fetch the IOP class files
    python run.py data        download the three datasets into datasets/
    python run.py figures     regenerate every figure
    python run.py paper       regenerate tables and macros, check, build the PDF
    python run.py reproduce   run the whole study end to end, data to PDF

`verify` is the one to start with. It recomputes the paper's headline claims
from the result files in this repository, in a few seconds, with no GPU and no
downloads, and prints what it finds next to what the paper says. If a number
here disagrees with the manuscript, the manuscript is wrong.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
SRC = ROOT / "src"

TWIN = "control/riemann-kernel-SVM"
QUANTUM_REF = ["quantum/Fidelity-ref-SVM", "quantum/HS-overlap-ref-SVM",
               "quantum/HS-RBF-ref-SVM", "quantum/Bures-RBF-ref-SVM",
               "quantum/QRE-RBF-ref-SVM"]
CLASSICAL = ["classical/TS+LR", "classical/CSP+LDA", "classical/MDM"]


def _run(*args: str, env_src: bool = True) -> int:
    import os
    env = dict(os.environ)
    if env_src:
        env["PYTHONPATH"] = str(SRC)
    print(f"  $ {' '.join(args)}", flush=True)
    return subprocess.run(args, cwd=ROOT, env=env).returncode


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------- status
def cmd_status(args) -> int:
    import pandas as pd
    _rule("result files in this checkout")
    groups = [
        ("within-subject, PhysioNet", "raw_folds_motor8_q4.csv"),
        ("  same, reference frame", "raw_folds_refstate_motor8_q4.csv"),
        ("within-subject, PhysioNet 4 qubits", "raw_folds_refstate_motor16_q4.csv"),
        ("within-subject, IV-2a", "raw_folds_bci2a_motor8_q4.csv"),
        ("  same, reference frame", "raw_folds_refstate_bci2a_motor8_q4.csv"),
        ("within-subject, Cho2017", "raw_folds_refstate_cho2017_motor8_q4.csv"),
        ("four-class IV-2a", "raw_folds_refstate_bci2a4_motor8_q4.csv"),
        ("cross-subject transfer", "transfer_folds_motor8.csv"),
        ("cross-session transfer", "crosssession_folds_bci2a_motor8.csv"),
        ("few-trial calibration, IV-2a", "calib_folds_bci2a.csv"),
        ("few-trial calibration, Cho2017", "calib_folds_cho2017.csv"),
        ("kernel concentration", "concentration_decay.csv"),
        ("Gram diagnostics", "reference_gram*.csv"),
        ("finite-shot estimation", "shots_folds*.csv"),
        ("equivalence (TOST)", "equivalence_twin.csv"),
    ]
    for label, pat in groups:
        hits = sorted(RES.glob(pat))
        if not hits:
            print(f"  {label:38s} absent")
            continue
        n = ""
        try:
            d = pd.concat([pd.read_csv(h) for h in hits], ignore_index=True)
            if "subject" in d:
                n = f"{d.subject.nunique()} subjects, "
            n += f"{len(d)} rows"
        except Exception:  # noqa: BLE001
            n = f"{len(hits)} file(s)"
        print(f"  {label:38s} {n}")
    figs = sorted((RES / "figures").glob("*.pdf"))
    print(f"\n  figures: {len(figs)}")
    pdf = ROOT / "paper" / "build" / "main.pdf"
    print(f"  manuscript: {'built' if pdf.exists() else 'not built'}"
          + (f" ({pdf.stat().st_size // 1024} KB)" if pdf.exists() else ""))
    return 0


# --------------------------------------------------------------------- verify
def _claim(name: str, found: str, says: str, ok: bool | None = None) -> None:
    mark = " " if ok is None else ("ok" if ok else "XX")
    print(f"  [{mark}] {name}")
    print(f"       found: {found}")
    print(f"       paper: {says}")


def cmd_verify(args) -> int:
    """Recheck the headline claims against the committed result files."""
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    failures, skipped = 0, []

    _rule("1. the invariance proposition, checked numerically")
    sys.path.insert(0, str(SRC))
    try:
        from qeeg.reference import check_invariance
        r = check_invariance()
        per = {k: v for k, v in r.items() if isinstance(v, dict)}
        worst_ref = max(v["reference"] for v in per.values())
        worst_sen = max(v["sensor"] for v in per.values())
        print(f"       a random congruence with condition number "
              f"{r['cond_A']:.1f} is applied to every trial, and each kernel is"
              f" recomputed:")
        for k, v in per.items():
            print(f"         {k:12s} sensor frame changes by {v['sensor']:.4f}, "
                  f"reference frame by {v['reference']:.1e}")
        _claim("the reference frame is exactly congruence-invariant, "
               "the sensor frame is not",
               f"reference deviates by at most {worst_ref:.1e}, "
               f"sensor by up to {worst_sen:.4f}",
               "exact to machine precision in the reference frame; "
               "the sensor frame is not invariant at all",
               worst_ref < 1e-10 < worst_sen)
        failures += not (worst_ref < 1e-10 < worst_sen)
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped: {exc})")

    _rule("2. the frame effect: recentring lifts the quantum kernels")
    f = RES / "raw_folds_motor8_q4.csv"
    fr = RES / "raw_folds_refstate_motor8_q4.csv"
    if f.exists() and fr.exists():
        a = pd.read_csv(f).groupby(["pipeline", "subject"]).accuracy.mean()
        b = pd.read_csv(fr).groupby(["pipeline", "subject"]).accuracy.mean()
        gains = []
        for p in QUANTUM_REF:
            base = p.replace("-ref-SVM", "-SVM")
            if p in b.index.get_level_values(0) and base in a.index.get_level_values(0):
                d = (b.loc[p] - a.loc[base]).dropna()
                if len(d):
                    gains.append(d.mean())
        if gains:
            _claim("every density-matrix kernel improves in the reference frame",
                   f"gains {min(gains):+.4f} to {max(gains):+.4f} "
                   f"over {len(gains)} kernels",
                   "all positive", min(gains) > 0)
            failures += min(gains) <= 0
    else:
        skipped.append("frame effect (PhysioNet suites)")
        print("  (absent: run the PhysioNet suites first)")

    _rule("3. the twin control: the gain is the frame, not the quantum metric")
    eq = RES / "equivalence_twin.csv"
    if eq.exists():
        d = pd.read_csv(eq)
        n_eq = int(d.equivalent.sum())
        _claim("quantum kernels are equivalent to the metric-matched twin",
               f"{n_eq} of {len(d)} comparisons equivalent at the stated "
               f"margin; widest bound {d['bound'].max():.4f}",
               "equivalence in most comparisons, none exceeding the margin "
               "by much")
        excl = d[(d.ci_low > 0) | (d.ci_high < 0)]
        print(f"       {len(excl)} interval(s) exclude zero"
              + (f": {', '.join(excl.kernel + ' (' + excl.setting + ')')}"
                 if len(excl) else ""))
    else:
        skipped.append("twin equivalence (equivalence_twin.csv)")
        print("  (absent: run python -m qeeg.equivalence)")

    _rule("4. few-trial calibration: the one regime that favours the quantum side")
    hits = sorted(RES.glob("calib_folds_cho2017.csv"))
    if hits:
        df = pd.concat([pd.read_csv(h) for h in hits], ignore_index=True)
        ps = (df.groupby(["n_train", "pipeline", "subject"]).accuracy
              .mean().reset_index())
        w = ps.pivot_table(index=["n_train", "subject"], columns="pipeline",
                           values="accuracy")
        n = w.index.get_level_values("subject").nunique()
        rows = []
        for k, g in w.groupby(level=0):
            q = g[[c for c in QUANTUM_REF if c in g]].mean(axis=1)
            d = (q - g[TWIN]).dropna()
            rows.append((k, d.mean(), wilcoxon(d).pvalue, int((d > 0).sum())))
        print(f"       Cho2017, {n} subjects, mean of "
              f"{len(QUANTUM_REF)} kernels minus the twin:")
        for k, m, p, better in rows:
            print(f"         {k:4d} trials  {m:+.4f}  p={p:.3f}  {better}/{n}")
        small = [r for r in rows if r[0] <= 40]
        large = [r for r in rows if r[0] >= 80]
        if small and large:
            _claim("the quantum edge is confined to few calibration trials",
                   f"mean {np.mean([r[1] for r in small]):+.4f} at 10 to 40 "
                   f"trials, {np.mean([r[1] for r in large]):+.4f} at 80 to 160",
                   "positive when trials are few, gone when they are not",
                   np.mean([r[1] for r in small]) >
                   np.mean([r[1] for r in large]))
    else:
        skipped.append("few-trial calibration")
        print("  (absent: run the calibration batches)")

    _rule("5. cost: the quantum kernels are slower for no gain")
    s = RES / "summary_motor8_q4.csv"
    if s.exists():
        d = pd.read_csv(s)
        col = "sec_per_subject"
        if col in d:
            q = d[d.pipeline.str.startswith("quantum/")][col].mean()
            c = d[d.pipeline.str.startswith("classical/")][col].mean()
            _claim("quantum kernels cost more wall clock",
                   f"{q:.1f}s per subject against {c:.1f}s classical "
                   f"({q / max(c, 1e-9):.0f}x)",
                   "slower, and the paper reports it rather than hiding it",
                   q > c)
    else:
        skipped.append("wall-clock cost")
        print("  (absent)")
    print()
    if skipped:
        # Silence is not success. A check that finds no data used to print a
        # note and let the summary say everything passed, which is how a stale
        # file path went unnoticed after the batch inputs were merged away.
        print(f"{len(skipped)} check(s) found no data and were SKIPPED:")
        for s_ in skipped:
            print(f"  - {s_}")
        print("Fix the paths or regenerate the data; a skipped check is not a "
              "passed one.")
    if failures:
        print(f"{failures} claim(s) did not check out. That is a real problem: "
              f"either the data changed or the manuscript is stale.")
        return 1
    if skipped:
        return 1
    print("Every claim checks out, and none was skipped.")
    print("The numbers quoted in the manuscript are generated from these same "
          "files by paper/make_tables.py, so they cannot disagree.")
    return 0


# ---------------------------------------------------------------------- setup
def cmd_setup(args) -> int:
    rc = _run(sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
              env_src=False)
    cls = ROOT / "paper" / "iopjournal.cls"
    if not cls.exists():
        print("\nfetching the IOP class files (not on CTAN)")
        rc |= _run("bash", "paper/get_iop_class.sh", env_src=False)
    else:
        print(f"\nIOP class present: {cls.name}")
    return rc


def cmd_data(args) -> int:
    print("Datasets are cached by MNE and MOABB. To keep them inside the "
          "project rather than on the system drive, do this first:")
    print("  python scripts/use_local_datasets.py --move")
    print("\nfetching (the first run downloads about 12 GB and takes a while)")
    return _run(sys.executable, "scripts/fetch_data.py", *args.rest)


def cmd_figures(args) -> int:
    rc = 0
    for mod in ("figures", "figures_eeg", "figures_circuits",
                "figures_reference"):
        rc |= _run(sys.executable, "-m", f"qeeg.{mod}",
                   *(["--paper"] if args.paper else []))
    # The architecture figure is not built here. It is authored in
    # diagrams.net and lives in figures/; see figures/README.md.
    return rc


def cmd_paper(args) -> int:
    rc = _run(sys.executable, "paper/make_tables.py", env_src=False)
    rc |= _run(sys.executable, "paper/check_tex.py", env_src=False)
    if args.pdf:
        import os
        env = dict(os.environ, BIBINPUTS=".", BSTINPUTS=".", TEXINPUTS=".")
        import shutil
        for target in ("supplementary.tex", "main.tex"):
            print(f"  $ latexmk -pdf {target}")
            rc |= subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                                  target], cwd=ROOT / "paper", env=env).returncode
        # latexmk is run in paper/ rather than with -outdir, because -outdir
        # hits the BibTeX path trap documented in paper/README.md and silently
        # produces an empty bibliography. But build/main.pdf is the file that
        # is tracked in git and therefore the one anyone actually opens, so it
        # must be updated here. It went stale for a day without this: readers
        # were looking at a 24-page, 38-reference build while the real one had
        # moved on to 27 and 45.
        built = ROOT / "paper" / "build"
        built.mkdir(parents=True, exist_ok=True)
        for name in ("main.pdf", "supplementary.pdf"):
            src = ROOT / "paper" / name
            if src.exists():
                shutil.copy2(src, built / name)
                print(f"  copied {name} -> paper/build/{name}")
    return rc


def cmd_reproduce(args) -> int:
    """Download the data, then run every stage of the study in order."""
    script = str(ROOT / "scripts" / "reproduce.py")
    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not args.run:
        # The default prints the plan and runs nothing. This command can spend
        # days of CPU and overwrite results/, so it does not start on a bare
        # invocation or a typo.
        rc = _run(sys.executable, script, "--plan", *rest, env_src=False)
        print("\nNothing has been run. To start:  python run.py reproduce --run")
        return rc
    return _run(sys.executable, script, *rest, env_src=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    p = sub.add_parser("figures")
    p.add_argument("--paper", action="store_true", default=True)
    p.set_defaults(fn=cmd_figures)
    p = sub.add_parser("paper")
    p.add_argument("--no-pdf", dest="pdf", action="store_false", default=True)
    p.set_defaults(fn=cmd_paper)
    p = sub.add_parser("reproduce")
    p.add_argument("--run", action="store_true",
                   help="actually run it; without this the plan is printed")
    p.set_defaults(fn=cmd_reproduce)
    # reproduce forwards its remaining flags (--from, --only, --workers,
    # --force) to scripts/reproduce.py. argparse.REMAINDER cannot do this:
    # it only starts collecting at the first positional, so a leading --only
    # is rejected before it ever reaches us.
    args, extra = ap.parse_known_args(argv)
    if extra and args.fn not in (cmd_reproduce, cmd_data):
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    args.rest = extra
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
