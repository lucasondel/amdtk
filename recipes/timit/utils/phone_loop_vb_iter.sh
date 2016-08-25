#!/usr/bin/env bash

if [ $# -ne 6 ]; then
    echo "usage: $0 <setup.sh> <parallel_opts> <keys> <ac_weight> <model_in_dir> <model_out_dir> "
    echo "                                                                                       "
    echo "Run one iteration of the VB training for the phone-loop.                               "
    echo "                                                                                       "
    exit 0 
fi

setup="$1"
parallel_opts="$2"
keys="$3"
ac_weight="$4"
model="$5/model.bin"
out_dir="$6"

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
        "amdtk_ploop_exp --ac_weight=$ac_weight $model \
            $fea_dir/\$ITEM1.$fea_ext $out_dir/\$ITEM1.acc" \
        "$out_dir" || exit 1

    # Accumulate the statistics. This step could be further
    # optimized by parallelizing the accumulation.
    find "$out_dir" -name "*.acc" > "$out_dir/stats.list" || exit 1
    amdtk_ploop_acc "$out_dir/stats.list" "$out_dir/total_acc_stats" || exit 1

    # VB M-step: from the sufficient statistics we update the
    # parameters of the posteriors.
    llh=$(amdtk_ploop_max "$model" \
        "$out_dir/total_acc_stats" \
        "$out_dir/model.bin" || exit 1)

    # Keep track of the lower bound on the log-likelihood.
    echo "$llh" > "$out_dir/llh.txt" || exit 1

    # Clean up the stats file as they can take a lot of space.
    rm "$out_dir/stats.list" || exit 1
    find "$out_dir/" -name "*.acc" -exec rm {} + || exit 1

    date > "$out_dir/.done"
else
    echo "Iteration has already been done. Skipping."
fi

