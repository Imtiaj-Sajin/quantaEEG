#!/usr/bin/env bash
# Fetch the IOP Publishing LaTeX class files needed to build main.tex.
#
# This pulls IOP's OWN package, not a mirror:
#
#   https://publishingsupport.iopscience.iop.org/questions/latex-template/
#     -> ioplatextemplate.zip   (package dated 2025/07)
#
# The zip contains iopjournal.cls [2024/01/31], the class IOP currently
# distributes, plus orcid.pdf (the little ORCID glyph that \orcid{} embeds),
# a template .tex and the guidelines PDF.
#
# History worth knowing: this repo previously built against `iopart.cls`,
# fetched from a third-party GitHub mirror. `iopart` is the legacy class and
# is NOT in IOP's current package at all, so there was nothing official left
# to check that mirror against. The manuscript was migrated to `iopjournal`
# on 2026-09-05.
#
# These files are third-party and are NOT committed to this repository.
#
#   bash paper/get_iop_class.sh

set -euo pipefail
cd "$(dirname "$0")"

ZIP="https://publishingsupport.iopscience.iop.org/wp-content/uploads/2025/07/ioplatextemplate.zip"
CTAN="http://mirrors.ctan.org/biblio/bibtex/contrib/iopart-num"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching IOP's official LaTeX package ..."
curl -fsSL --max-time 120 -o "$tmp/iop.zip" "$ZIP"
printf '  %-26s %8s bytes\n' "ioplatextemplate.zip" "$(wc -c < "$tmp/iop.zip")"

# Only the two files the build needs; the guidelines PDF and the sample
# template are useful to read but do not belong next to the manuscript.
python -c "
import zipfile, sys, shutil, os
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
for want in ('iopjournal.cls', 'orcid.pdf'):
    match = [n for n in names if n.endswith(want)]
    if not match:
        sys.exit('MISSING from IOP package: ' + want)
    with z.open(match[0]) as src, open(want, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    print('  %-26s %8d bytes' % (want, os.path.getsize(want)))
" "$tmp/iop.zip"

# Confirm we really got IOP's class and not something that merely unzipped.
if ! grep -q 'ProvidesClass{iopjournal}' iopjournal.cls; then
    echo "ERROR: iopjournal.cls does not declare itself; refusing." >&2
    exit 1
fi
grep -m1 'ProvidesClass{iopjournal}' iopjournal.cls | sed 's/^/  version: /'
grep -m1 'Copyright' iopjournal.cls | sed 's/^ *%* */  /'

# iopart-num.bst is IOP's numeric bibliography style and IS on CTAN.
curl -fsSL --max-time 60 -o iopart-num.bst "$CTAN/iopart-num.bst"
printf '  %-26s %8s bytes\n' "iopart-num.bst" "$(wc -c < iopart-num.bst)"

echo
echo "Done. Build with:"
echo "  cd paper && latexmk -pdf -interaction=nonstopmode -outdir=build main.tex"
echo "(see README for the BIBINPUTS/TEXINPUTS exports the -outdir build needs)"
