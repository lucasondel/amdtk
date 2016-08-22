#!/usr/bin/env bash

if [ $# -ne 3 ]; then
    echo ""
    echo "Estimate the hierarchical Pitman-Yor language model."
    echo ""
    echo "usage: $0 <setup.sh> <labels_dir> <out_dir>"
    exit 1
fi

setup=$1
text="$2/text"
out_dir="$3"

source $setup || exit 1

if [ ! -e "$out_dir/.done" ]; then
    mkdir -p "$out_dir"

    # Generate the vocabulary.
    vocab_file="$out_dir/vocabulary.txt"
    for i in $(seq $truncation); do
        echo "a$i"
    done > "$vocab_file"
    echo "sil" >> "$vocab_file"

    # Carry out lm training
    amdtk_lm_create \
        "$lm_params" \
        "$vocab_file" \
        "$text" \
        "$out_dir/lm.bin" \
        2>&1 > "$out_dir/amdtk_lm_create.log"

    date > "$out_dir/.done"
else
    echo "The model has already been created. Skipping."
fi
