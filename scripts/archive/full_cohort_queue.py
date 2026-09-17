"""Write the job list for rerunning every PhysioNet within-subject suite on all
104 eligible subjects, in an order that keeps the CPU busy while data download.

    python scripts/full_cohort_queue.py > results/full_cohort_jobs.txt
    cat results/full_cohort_jobs.txt | xargs -P 10 -I{} bash -c "{}"

EEGMMIDB has 109 subjects; 88, 89, 92, 100 and 104 have documented recording
defects and are excluded, leaving 104. They are split into ten batches, and
the first three batches (subjects 1 to 32, already cached) of every suite are
queued before any batch that still waits on a download.
"""
from __future__ import annotations

BAD = {88, 89, 92, 100, 104}
SUBJECTS = [s for s in range(1, 110) if s not in BAD]
N_BATCHES = 10

# (tag prefix, extra qeeg.benchmark arguments). Tags are merged later into the
# canonical file names the manuscript scripts read.
SUITES = [
    ("all_core", ""),
    ("all_ext", "--suite extended"),
    ("all_16ch", "--suite extended --channels motor16"),
    ("all_ext_s1", "--suite extended --seed 1"),
    ("all_ext_s2", "--suite extended --seed 2"),
    ("all_fb", "--suite filterbank"),
]


def batches():
    size = -(-len(SUBJECTS) // N_BATCHES)
    return [SUBJECTS[i:i + size] for i in range(0, len(SUBJECTS), size)]


def main():
    bs = batches()
    assert sum(len(b) for b in bs) == 104
    cached = [i for i, b in enumerate(bs) if max(b) <= 32]
    later = [i for i in range(len(bs)) if i not in cached]
    for group in (cached, later):
        for prefix, extra in SUITES:
            for i in group:
                subs = ",".join(str(s) for s in bs[i])
                print(f"bash scripts/cohort_job.sh {prefix}_b{i + 1:02d} {subs} {extra}".rstrip())


if __name__ == "__main__":
    main()
