#!/usr/bin/env bash

#
# Estimate the hierarchical Pitman-Yor language model.
#

if [ $# -ne 1 ]; then
    echo "usage: $0 <setup.sh>"
    exit 1
fi

setup=$1

source $setup || exit 1

if [ ! -e "$lm_base_path/.done" ]; then
    mkdir -p "$lm_base_path"

    # Generate the vocabulary.
    vocab_file="$lm_base_path/vocabulary.txt"
    for i in $(seq $truncation); do
        echo "a$i"
    done > "$vocab_file"

    # Carry out lm training
    amdtk_lm_create \
        $fixed \
        "$params" \
        "$vocab_file" \
        "$text_input_file" \
        "$lm_output_file" \
        "$symbols_file" \
        2>&1 > "$lm_base_path/amdtk_lm_create.log"

    date > "$lm_base_path/.done"
else
    echo "The model has already been created. Skipping."
fi
