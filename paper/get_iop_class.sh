#!/usr/bin/env bash
# Fetch the IOP Publishing LaTeX class files needed to build main.tex.
#
# iopart.cls is distributed by IOP for authors and is NOT on CTAN, so it is
# not bundled with TeX Live, MiKTeX or Tectonic. It is therefore not committed
# to this repository either (it is third-party); run this script once before
# the first build.
#
# For an actual submission, prefer the official copy from IOP:
#   https://publishingsupport.iopscience.iop.org/questions/latex-template/
# The mirror used here is convenient for local builds and is byte-identical
# to the version IOP ships at the time of writing.
#
#   bash paper/get_iop_class.sh

set -euo pipefail
cd "$(dirname "$0")"

MIRROR="https://raw.githubusercontent.com/etgroup/iop-latex-template/main"
CTAN="http://mirrors.ctan.org/biblio/bibtex/contrib/iopart-num"

echo "Fetching IOP class files into $(pwd) ..."
for f in iopart.cls iopart10.clo iopart12.clo iopams.sty setstack.sty harvard.sty; do
    curl -fsSL --max-time 60 -o "$f" "$MIRROR/$f"
    printf '  %-16s %8s bytes\n' "$f" "$(wc -c < "$f")"
done

# iopart-num.bst (the Harvard-numeric bibliography style) IS on CTAN.
curl -fsSL --max-time 60 -o iopart-num.bst "$CTAN/iopart-num.bst"
printf '  %-16s %8s bytes\n' "iopart-num.bst" "$(wc -c < iopart-num.bst)"

echo "Done. Now build with:  tectonic -X compile main.tex --outdir build"
