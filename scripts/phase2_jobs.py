"""Job list for revision phase 2, run after the full-cohort benchmark queue.

    python scripts/phase2_jobs.py > results/phase2_jobs.txt
    cat results/phase2_jobs.txt | xargs -P 10 -I{} bash -c "{}"

Full-cohort PhysioNet analyses that are not benchmark batches (Gram
diagnostics at four register sizes, the concentration sweep, finite shots and
leave-one-subject-out transfer) write to results/full104/, a staging folder, so
the 30-subject files the manuscript currently reads are not overwritten until
the new ones are checked. The two new experiments (few-trial calibration, and
four-class IV-2a) write straight to results/ under new names. Every job skips
itself if its output exists, so the list can be restarted.

Transfer is split into six held-out chunks, queued first: it is the longest job
and the only memory-heavy one (about 1.3 GB per process at 104 subjects).
"""
from __future__ import annotations

BAD = {88, 89, 92, 100, 104}
PHYSIONET = [s for s in range(1, 110) if s not in BAD]
ENV = "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src"
STAGE = "results/full104"


def chunks(xs, n):
    size = -(-len(xs) // n)
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def job(tag, done_file, command):
    log = f"results/run_{tag}.log"
    return (f"cd \"$(git rev-parse --show-toplevel)\" && mkdir -p {STAGE} && "
            f"if [ -f {done_file} ]; then echo skip {tag}; else {ENV}; "
            f"{command} > {log} 2>&1; echo JOB_EXIT=$? >> {log}; fi")


def main():
    lines = []
    # 1. Transfer, 104 subjects, held-out subjects split over six processes.
    for i, part in enumerate(chunks(PHYSIONET, 6), 1):
        tag = f"tr104_c{i:02d}"
        held = ",".join(map(str, part))
        lines.append(job(tag, f"{STAGE}/transfer_folds_{tag}.csv",
                         f"python -u -m qeeg.transfer --subjects 109 --start 1 "
                         f"--heldout {held} --out {STAGE} --tag {tag}"))
    # 2. Gram diagnostics at every register size.
    for ch in ("motor8", "motor16", "motor32", "all64"):
        out = f"{STAGE}/reference_gram_{ch}.csv"
        lines.append(job(f"gram104_{ch}", out,
                         f"python -u -m qeeg.reference --gram --subjects 109 --start 1 "
                         f"--channels {ch} --out {out}"))
    # 3. Concentration sweep and finite shots.
    lines.append(job("conc104", f"{STAGE}/concentration_decay.csv",
                     f"python -u -m qeeg.concentration --subjects 109 --start 1 --out {STAGE}"))
    lines.append(job("shots104", f"{STAGE}/shots_folds_motor8.csv",
                     f"python -u -m qeeg.shots --subjects 109 --start 1 --out {STAGE}"))
    # 4. Few-trial calibration: Cho2017 (52 subjects) and IV-2a (9 subjects).
    for i, part in enumerate(chunks(list(range(1, 53)), 8), 1):
        tag = f"calib_cho_b{i:02d}"
        lines.append(job(tag, f"results/calib_folds_{tag}.csv",
                         f"python -u -m qeeg.calibration --dataset cho2017 "
                         f"--subject-list {','.join(map(str, part))} --tag {tag}"))
    for s in range(1, 10):
        tag = f"calib_bci_b{s:02d}"
        lines.append(job(tag, f"results/calib_folds_{tag}.csv",
                         f"python -u -m qeeg.calibration --dataset bci2a "
                         f"--subject-list {s} --tag {tag}"))
    # 5. Four-class IV-2a, extended suite, one subject per process.
    for s in range(1, 10):
        tag = f"bci4_b{s:02d}"
        lines.append(job(tag, f"results/raw_folds_{tag}.csv",
                         f"python -u -m qeeg.benchmark --dataset bci2a --classes 4 "
                         f"--suite extended --subject-list {s} --tag {tag} --no-stats --resume"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
