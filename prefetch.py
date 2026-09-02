"""Parallel prefetch of EEGMMIDB files into the MNE cache.

MNE/pooch downloads sequentially; the benchmark spends most of its wall clock
waiting on the network. This warms the same cache with a thread pool, starting
well ahead of where the sequential run currently is so the two never contend
for the same file.
"""
import sys, warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")
from mne.datasets import eegbci

start, stop = int(sys.argv[1]), int(sys.argv[2])
SUBJECTS = [s for s in range(start, stop + 1) if s not in (88, 89, 92, 100, 104)]

def grab(s):
    try:
        eegbci.load_data(subjects=s, runs=[4, 8, 12], update_path=True)
        return f"ok S{s:03d}"
    except Exception as e:
        return f"FAIL S{s:03d}: {type(e).__name__}"

with ThreadPoolExecutor(max_workers=8) as pool:
    for msg in pool.map(grab, SUBJECTS):
        print(msg, flush=True)
print("prefetch done")
