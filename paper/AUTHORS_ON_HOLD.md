# Co-authors on hold

Two co-authors were removed from the manuscript on 2026-09-17, before
submission, because they had not yet confirmed that they want to be authors.
Nothing about their contribution was lost: this file records everything needed
to restore them exactly as they were. If either confirms, follow "How to
restore" below.

## Who

| | Md Wahiduzzaman Suva | Esm E Moula Chowdhury Abha |
|---|---|---|
| Position in the author list | 2nd (after Sajin) | 3rd (after Suva) |
| Name as printed | Md Wahiduzzaman Suva | Esm E Moula Chowdhury Abha |
| Initials in the e-mail block | M W Suva | E E M C Abha |
| Affiliation | 1: Department of Computer Science, American International University-Bangladesh (AIUB), Dhaka, Bangladesh | same |
| E-mail | 26-94088-2@student.aiub.edu | 26-94089-2@student.aiub.edu |
| ORCID | none on record | none on record |

## Their contributions (CRediT roles)

Set by the corresponding author on 2026-09-15.

- **Md Wahiduzzaman Suva**: investigation, resources, validation, writing
  (review and editing). Shared computing resources for the benchmark runs
  (resources, together with Sajin), carried out validation, revised the
  writing, and had investigation added later the same day.
- **Esm E Moula Chowdhury Abha**: software, visualisation, writing (review
  and editing). Visualisation was moved to Abha from Sajin, software was added
  later the same day, and Abha revised the writing.

With both removed, no remaining author holds the **validation** or
**visualisation** role. If they stay off the paper, the corresponding author
should decide whether those roles belong to someone else on the list.

## The full author order before removal

Sajin, Suva, Abha, Saif, Chayon (supervisor last), set on 2026-09-15.

After removal: Sajin, Saif, Chayon.

## How to restore

In `paper/main.tex` and `paper/supplementary.tex`, the `\author{}` block was:

```latex
\author{Md.~Imtiaj~Alam~Sajin$^{1,*}$\orcid{0009-0009-2423-1835},
Md~Wahiduzzaman~Suva$^{1}$,
Esm~E~Moula~Chowdhury~Abha$^{1}$,
Samiul~Islam~Saif$^{1}$\orcid{0009-0009-7116-7151} and
Muhammad~Hasibur~Rashid~Chayon$^{1}$\orcid{0000-0003-3741-6651}}
```

In `paper/main.tex`, the `\email{}` block was:

```latex
\email{26-94090-2@student.aiub.edu, imtiajsajin@gmail.com (M I A Sajin);
26-94088-2@student.aiub.edu (M W Suva); 26-94089-2@student.aiub.edu
(E E M C Abha); Md.samiulislam.aiub@gmail.com (S I Saif); chayon@aiub.edu
(M H R Chayon)}
```

and the `\roles{}` block was:

```latex
\roles{\textbf{Md. Imtiaj Alam Sajin}: conceptualisation, data curation,
formal analysis, investigation, methodology, resources, software, writing
(original draft). \textbf{Md Wahiduzzaman Suva}: investigation, resources,
validation, writing (review and editing). \textbf{Esm E Moula Chowdhury
Abha}: software, visualisation, writing (review and editing).
\textbf{Samiul Islam Saif}:
investigation, software, writing (review and editing). \textbf{Muhammad
Hasibur Rashid Chayon}: supervision, writing (review and editing).}
```

Paste those back, rebuild (`python paper/make_tables.py`, compile the
supplement then the manuscript), rebuild `submission.zip`, and update the
author order in `CLAUDE.md` and `paper/README.md`. Git history also holds the
last version with all five authors: commit `803612b` and earlier.
