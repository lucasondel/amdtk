#!/usr/bin/env bash

if [ $# -ne 3 ]; then
    echo ""
    echo "Create and initialize an infinite phone-loop model."
    echo ""
    echo "usage: $0 <setup.sh> <keys> <outdir>"
    exit 1
fi

setup="$1"
keys="$2"
out_dir="$3"

source $setup || exit 1

if [ ! -e "$out_dir/.done" ]; then
    mkdir "$out_dir"

    # Create the list of features files for the train set.
    while read line; do
        echo "$fea_dir/$line.$fea_ext" >> "$out_dir"/fea.list
    done < "$keys"

    # Get the mean and covariance of the data to initialize the model.
    # For huge database it is probably better to estimate these
    # statistics on a part of the database.
    amdtk_ploop_stats "$out_dir/fea.list" "$out_dir/stats"

    # Create the model. All the hyper-paramaters are identical for each
    # Gaussian of the phone-loop but the hyper-parameters of the mean.
    # These means are sampled from the a Gaussian distribution with mean
    # and covariance of the database.
    amdtk_ploop_create \
        --sil_ngauss "$sil_ngauss" \
        --concentration "$concentration" \
        --truncation "$truncation" \
        --nstates "$nstates" \
        --ncomponents "$ncomponents" \
        --alpha "$alpha" \
        --kappa "$kappa" \
        --a "$a" \
        --b "$b" 

        "$out_dir/stats" "$out_dir/model.bin"

    date > "$out_dir/.done"
else
    echo "The model has already been created. Skipping."
fi

