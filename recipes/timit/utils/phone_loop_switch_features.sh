#!/usr/bin/env bash

if [ $# -ne 7 ]; then
    echo ""
    echo "Initialize a phone loop model on a new set of features."
    echo ""
    echo "usage: $0 <setup.sh> <parallel_opts> <keys>  <model_dir> <new_fea_dir> <new_model_dir> <out_dir>"
    exit 1
fi

setup="$1"
parallel_opts="$2"
keys="$3"
model="$4/model.bin"
new_fea_dir="$5"
new_model="$6/model.bin"
out_dir="$7"

source $setup || exit 1

if [ ! -e $out_dir/.done ]; then
    mkdir -p "$out_dir"
    
    # VB E-step: estimate the posterior distribution of the
    # latent variables.
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$parallel_opts" \
        "pl-vbexp" \
        "$keys" \
        "amdtk_ploop_exp --new_feats=$new_fea_dir/\$ITEM1.$fea_ext \
            $model $fea_dir/\$ITEM1.$fea_ext \
            $out_dir/\$ITEM1.acc" \
        "$out_dir" || exit 1

    # Accumulate the statistics. This step could be further
    # optimized by parallelizing the accumulation.
    find "$out_dir" -name "*.acc" > \
        "$out_dir/stats.list" || exit 1
    amdtk_ploop_acc "$out_dir/stats.list" \
        "$out_dir/total_acc_stats" || exit 1

    # VB M-step: from the sufficient statistics we update the
    # parameters of the posteriors.
    llh=$(amdtk_ploop_max "$new_model" \
        "$out_dir/total_acc_stats" \
        "$out_dir/model.bin" || exit 1)

    # Clean up the statistics as they can take a lot of space.
    rm "$out_dir/stats.list" || exit 1
    find "$out_dir/" -name "*.acc" -exec rm {} + || exit 1

    date > "$out_dir/.done"
else
    echo "The model has already been created. Skipping."
fi

