#!/usr/bin/env bash

#
# Sample paths from lattices.
#

if [ $# -ne 5 ]; then
    echo "usage: $0 <setup.sh> <ac_weight> <lm_weight> <lattice_dir> <out_dir>"
    exit 1
fi

setup=$1
ac_weight=$2
lm_weight=$3
latt_dir=$4
out_dir=$5

source "$setup" || exit 1

if [ ! -e "$out_dir"/.done ]; then
    # Create the output _directory.
    mkdir -p "$out_dir"

    # Sample a path for each lattice file.
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$latt_parallel_opts" \
        "sample-path" \
        "$train_keys" \
        "amdtk_lattice_sample_path --ac_weight=$ac_weight \
            --lm_weight=$lm_weight  $latt_dir/\$ITEM1.latt.gz \
            $out_dir/\$ITEM1.lab" \
        "$out_dir" || exit 1

    date > "$out_dir"/.done
else
    echo "The  paths have already been sampled. Skipping."
fi

