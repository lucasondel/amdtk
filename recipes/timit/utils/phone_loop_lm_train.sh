#!/usr/bin/env bash

#
# Train a HPYP language model on the best path from the AUD system.
#

if [ $# -ne 4 ]; then
    echo "usage: $0 <setup.sh> <niter> <lattices_dir> <out_dir>"
    exit 1
fi

setup="$1"
niter="$2"
latt_dir="$3"
out_dir="$4"

source "$setup" || exit 1

if [ ! -e "$out_dir/.done" ]; then

    # We first label the training data with the Viterbi algorithm.
    if [ ! -e "$out_dir"/sampled_path/.done ]; then

        # Create the output directory.
        mkdir -p "$out_dir"/sampled_paths

        # Sampling a path from the lattices.
        utils/phone_loop_sample_path.sh $setup $ac_weight $lm_weight \
            "$latt_dir" "$out_dir/sampled_paths" || exit 1

        # Create a text file from the HTK label files.
        labels_file=$(find "$out_dir"/sampled_paths -name "*.lab") || exit 1
        for f in $labels_file ; do
            cat "$f" | awk '{printf $3 " "} END {print ""}' \
                >> "$out_dir/sampled_paths/text" || exit 1
        done

        date > "$out_dir"/sampled_paths/.done
    else
        echo "The sampling has already been done. Skipping."
    fi

    # Train the model
    amdtk_lm_create --niter="$niter" --resample "$lm_params" "$out_dir"/text \
        "$out_dir"/lm.bin

    date > "$out_dir"/.done
else
    echo "The language model has already been trained. Skipping."
fi

