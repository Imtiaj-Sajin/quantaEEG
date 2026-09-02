# Carrying Claude Code context between machines

## What actually travels

The durable, machine-independent context is **in this repo**:

- `CLAUDE.md`: auto-loaded at the start of every Claude Code session, on any
  machine, at any path, with no setup.
- `RESEARCH.md`: the research record: verdict, literature, findings, strategy.

Clone the repo and a fresh session comes up oriented. This is the recommended
approach and needs nothing else.

## What does NOT travel

Conversation transcripts live in your **home directory**, not the project:

```
~/.claude/projects/<encoded-project-path>/<session-uuid>.jsonl
```

For this project on the original machine that was:

```
C:\Users\Imtiaj Sajin\.claude\projects\f--Imtiaj-Sajin-quantaEEG\
```

### The path-encoding trap

The folder name is derived from the project's **absolute path**, with `:`, `\`,
`/` and spaces each replaced by `-`:

| Project path | Folder name |
|---|---|
| `f:\Imtiaj Sajin\quantaEEG` | `f--Imtiaj-Sajin-quantaEEG` |
| `C:\Users\Imtiaj Sajin` | `C--Users-Imtiaj-Sajin` |
| `e:\Sajin work\FreshDevelopment\odin-ems` | `e--Sajin-work-FreshDevelopment-odin-ems` |

**Drive-letter case is preserved as typed**, both `F--Imtiaj-Sajin-boss-pdf`
and `f--Imtiaj-Sajin-quantaEEG` exist side by side. So a transcript copied to a
machine where the project sits at a different path will simply not be found,
and `claude --resume` will show nothing.

### If you do want the transcript

1. Copy the whole `<encoded-project-path>/` folder from the old machine's
   `~/.claude/projects/`.
2. On the new machine, place it under `~/.claude/projects/` renamed to match
   the new project path's encoding (or keep the project at the identical
   absolute path).
3. Run `claude --resume` in the project directory and pick the session.

Copying the `.jsonl` into the repo does **not** work, Claude Code only reads
transcripts from `~/.claude/projects/`, never from the project folder.

A caution: these transcripts are large (this project's is ~4.5 MB) and include
the entire conversation. Resuming a very long transcript is often *worse* than
starting fresh with a good `CLAUDE.md`. Prefer the repo-based approach unless
you specifically need the history.

---

## Setting up a second machine (what the repo does *not* carry)

Cloning gets you the code, the results and both markdown records. Four things
have to be rebuilt locally, and three of them are slow, so start them first and
read `RESEARCH.md` while they run.

### 1. Python packages

```bash
pip install -r requirements.txt
pip install -c constraints.txt moabb          # only for the BCI IV-2a runs
```

`constraints.txt` pins `scipy==1.15.3` and must be used for every install.
On the original Windows machine, scipy >= 1.16's `_batched_linalg` DLL is
blocked by Application Control and takes the whole stack down with an opaque
`ImportError`. Installing anything without `-c constraints.txt` risks a silent
upgrade.

If `pip` fails writing `Scripts\*.exe` ("could not install ... .deleteme"),
add `--user`; that puts the console scripts somewhere writable and the
libraries import the same either way.

### 2. Datasets (the slow one)

Nothing in `~/mne_data` travels, and it is not in the repo by design.

```bash
python -u prefetch.py 1 30      # PhysioNet EEGMMIDB, ~40 min for 30 subjects
```

BCI IV-2a comes down through MOABB the first time any `--dataset bci2a` run
touches it: 9 subjects, ~82 MB each, ~750 MB total. Warm it deliberately
rather than discovering it mid-benchmark:

```bash
PYTHONPATH=src python -u -c "import moabb.datasets as m; d=m.BNCI2014_001(); [d.get_data(subjects=[s]) for s in d.subject_list]"
```

### 3. LaTeX

`iopart.cls` is not on CTAN and is not committed (third-party). Fetch it once:

```bash
bash paper/get_iop_class.sh
```

Then build per `paper/README.md`, including the `BIBINPUTS`/`TEXINPUTS` exports
— without them BibTeX silently emits an empty bibliography and the failure
surfaces much later as a misleading `missing \item`.

### 4. Long runs: threads and batching

Two things learned the hard way on 12 cores:

- **Always set `OMP_NUM_THREADS=1`** (plus `OPENBLAS_`/`MKL_NUM_THREADS`) for
  benchmark runs. The matrices here are 8x8 to 45x45; BLAS threading is pure
  overhead, and five unconstrained processes spawned 23 threads each and spent
  most of their time context-switching.
- **Split long runs across processes**, not across threads:

```bash
for b in "01 1,2,3,4,5,6" "02 7,8,9,10,11,12"; do set -- $b
  OMP_NUM_THREADS=1 nohup env PYTHONPATH=src python -u -m qeeg.benchmark \
    --subject-list "$2" --suite filterbank --tag "fbbatch$1" --no-stats \
    > "results/run_fb$1.log" 2>&1 &
done
PYTHONPATH=src python -m qeeg.merge --pattern "raw_folds_fb*.csv" --tag filterbank_motor8
```

Each batch checkpoints to `raw_folds_<tag>.partial.csv` after every subject, so
a killed run loses at most one subject. `merge.py` de-duplicates on
`(subject, pipeline, fold)`, so re-running a batch is safe.

Verify processes with PowerShell `Get-CimInstance Win32_Process`, not
`tasklist`/`ps` under Git Bash — those have returned empty output unreliably
here, and two concurrent runs writing the same tag will corrupt each other.
