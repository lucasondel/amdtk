#!/usr/bin/env bash

if [ $# -ne 4 ]; then
    echo "usage: $0 <setup.sh> <lattice_file> <lm_fst_binary> <out_dir>"
    exit 1
fi

setup=$1
lattice_file=$2
lm_fst_binary=$3
out_dir=$4

source $setup || exit 1

# set prefix for wordend fst name
wordend_prefix="${rescoring_base_path}/wordend"

# set phi id for fstphicompose
phi_id=1

declare -a sym_args=("--isymbols=$rescoring_symbols_file" \
                     "--osymbols=$rescoring_symbols_file")

# convert from htk SLF to openfst text format if necessary
if [[ "$lattice_type" == "htk" ]]; then
    fstfile=$(mktemp)
    gunzip -c "$lattice_file" | htk_to_openfst.py - "$fstfile"
    out_lattice_file=${lattice_file/'.latt.gz'/'_P.fst.txt'}
else
    fstfile="$lattice_file"
    out_lattice_file=${lattice_file/'.fst.txt'/'_P.fst.txt'}
fi

# set proper directory for output file
out_lattice_file="$out_dir/$(basename $out_lattice_file)"

# create temporary file to use for best path and conversion to txt format
out_lattice_tempfile=$(mktemp)

label_outfile="$rescoring_labels_outdir/$(basename ${lattice_file/.*/".lab"})"
# - load the fst, insert word end markers
# - compose with the language model fst
# - remove wordend markers, merge paths again
# - and store the result in an fst.txt file
fstcompile ${sym_args[@]} "$fstfile" |
    fstarcsort --sort_type=olabel |
    fstcompose - "$wordend_prefix"_insert.fst |
    fstphicompose "$phi_id" - "$lm_fst_binary" |
    fstarcsort |
    fstcompose - "$wordend_prefix"_remove.fst |
    fstrmepsilon | fstdeterminize > "$out_lattice_tempfile"

# determine shortest path and convert to label file
fstshortestpath "$out_lattice_tempfile"|
    fstprint ${sym_args[@]} |
    shortest_path_to_labels.py - "$label_outfile"

# convert lattice to openfst txt format and zip
fstprint "${sym_args[@]}" "$out_lattice_tempfile" |
    gzip > "$out_lattice_file".gz

if [[ "$input" == "htk" ]]; then
    rm "$fstfile"
fi
rm "$out_lattice_tempfile"
