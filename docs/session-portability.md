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
