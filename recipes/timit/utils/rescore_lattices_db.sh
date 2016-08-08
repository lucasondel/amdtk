#!/usr/bin/env bash


if [ $# -ne 3 ]; then
    echo "usage: $0 <setup.sh> <lattice_dir> <out_dir>"
    exit 1
fi

setup=$1
lattice_dir=$2
out_dir=$3
source $setup || exit 1

# this is the filename of the ascii language model fst
# with ending as defined in amdtk_lm_create
lm_fst_txt="${rescoring_lm_file}.fst.txt"
# set filename for binary language model fst
lm_fst_binary="${rescoring_lm_file}.fst.binary"
# set prefix for wordend fst name
wordend_prefix="${rescoring_base_path}/wordend"

declare -a sym_args=("--isymbols=$rescoring_symbols_file" \
                     "--osymbols=$rescoring_symbols_file")

if [ ! -e "$out_dir/.done" ]; then
    mkdir -p "$out_dir"
    mkdir -p "$rescoring_labels_outdir"

    # take language model fst, compile it and perform an input arc sort
    fstcompile  "${sym_args[@]}" "$lm_fst_txt" |
        fstarcsort --sort_type=ilabel > "$lm_fst_binary"

    # build word-end fsts from given symbol table
    create_seq_end_fst.py "$rescoring_symbols_file" "$wordend_prefix"
    fstcompile ${sym_args[@]} "$wordend_prefix"_insert.fst.txt \
        > "$wordend_prefix"_insert.fst
    fstcompile ${sym_args[@]} "$wordend_prefix"_remove.fst.txt \
        > "$wordend_prefix"_remove.fst

    # create flist depending on lattice format
    if [[ "$lattice_type" == "htk" ]]; then
        suffix="*.latt.gz"
    else
        suffix="*.fst.txt"
    fi
    find "$lattice_dir"/ -name "$suffix" > "$out_dir"/lattices.list

    # call amdtk_run for rescore_lattices
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$latt_parallel_opts" \
        "rescore-lattices" \
        "$out_dir/lattices.list" \
        "utils/rescore_lattice.sh $setup \$ITEM1 $lm_fst_binary $out_dir" \
        "$out_dir" || exit 1

    date > "$out_dir/.done"
else
    echo "The lattices have already been rescored. Skipping"
fi


