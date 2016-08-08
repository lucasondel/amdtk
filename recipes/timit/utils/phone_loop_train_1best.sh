#!/usr/bin/env bash

#
# Train the inifinite phone-loop model.
#

if [ $# -ne 4 ]; then
    echo "usage: $0 <setup.sh> <model_dir> <labels_dir> <out_dir>"
    exit 1
fi

setup="$1"
model="$2/model.bin"
labels_dir="$3"
out_dir="$4"

source $setup || exit 1

if [ ! -e $out_dir/.done ]; then
    mkdir -p "$out_dir"

    # VB E-step: estimate the posterior distribution of the
    # latent variables.
    #amdtk_run $parallel_profile \
    #    --ntasks "$parallel_n_core" \
    #    --options "$train_parallel_opts" \
    #    "pl-vbexp" \
    #    "$train_keys" \
    #    "amdtk_ploop_1best_exp $model $labels_dir/\$ITEM1.lab \
    #        $fea_dir/\$ITEM1.$fea_ext $out_dir/\$ITEM1.acc" \
    #    "$out_dir"|| exit 1
    #
    ## Accumulate the statistics. This step could be further
    ## optimized by parallelizing the accumulation.
    #find "$out_dir" -name "*.acc" > "$out_dir/stats.list" || exit 1
    #amdtk_ploop_acc "$out_dir/stats.list" "$out_dir/total_acc_stats" \
    #    || exit 1

    # VB M-step: from the sufficient statistics we update the
    # parameters of the posteriors
    llh=$(amdtk_ploop_max $model "$out_dir/total_acc_stats" \
        "$out_dir/model.bin") || exit 1

    echo "log-likelihood >= $llh"

    # Keep track of the lower bound on the log-likelihood.
    echo "$llh" > "$out_dir"/llh.txt || exit 1

    # Clean up the statistics as they can take a lot of space.
    rm "$out_dir/stats.list" || exit 1
    find "$out_dir/" -name "*.acc" -exec rm {} + || exit 1

    date > "$out_dir/.done"
else
    echo "The model is already trained. Skipping."
fi

