#!/usr/bin/env bash

#
# Train the inifinite phone-loop model.
#

if [ $# -ne 4 ]; then
    echo "usage: $0 <setup.sh> <niter> <in_dir> <out_dir>"
    exit 1
fi

setup="$1"
niter="$2"
init_model="$3/model.bin"
out_dir="$4"

source $setup || exit 1

if [ ! -e $out_dir/.done ]; then
    mkdir -p "$out_dir"
    
    # First, we copy the inital model to the iteration 0 directory.
    if [ ! -e "$out_dir"/iter0/.done ]; then
        mkdir "$out_dir"/iter0
        cp "$init_model" "$out_dir/iter0/model.bin"  

        date > "$out_dir"/iter0/.done
    else
        echo "The initial model has already been set. Skipping."
    fi

    # Now start to (re-)train the model. 
    for i in $(seq "$niter") ; do
        if [ ! -e "$out_dir/iter$i/.done" ]; then
            mkdir "$out_dir/iter$i" || exit 1

            # We pick up the model from the last iteration.
            model="$out_dir/iter$((i-1))/model.bin" || exit 1

            # VB E-step: estimate the posterior distribution of the
            # latent variables.
            amdtk_run $parallel_profile \
                --ntasks "$parallel_n_core" \
                --options "$train_parallel_opts" \
                "pl-vbexp" \
                "$train_keys" \
                "amdtk_ploop_exp $model $fea_dir/\$ITEM1.$fea_ext \
                $out_dir/iter$i/\$ITEM1.acc" \
                "$out_dir/iter$i" || exit 1

            # Accumulate the statistics. This step could be further
            # optimized by parallelizing the accumulation.
            find "$out_dir/iter$i" -name "*.acc" > \
                "$out_dir/iter$i/stats.list" || exit 1
            amdtk_ploop_acc "$out_dir/iter$i/stats.list" \
                "$out_dir/iter$i/total_acc_stats" || exit 1

            # VB M-step: from the sufficient statistics we update the
            # parameters of the posteriors.
            llh=$(amdtk_ploop_max "$model" \
                "$out_dir/iter$i/total_acc_stats" \
                "$out_dir/iter$i/model.bin" || exit 1)

            echo "Iteration: $i log-likelihood >= $llh"|| exit 1

            # Keep track of the lower bound on the log-likelihood.
            echo "$llh" > "$out_dir"/iter"$i"/llh.txt || exit 1

            # Clean up the statistics as they can take a lot of space.
            rm "$out_dir/iter$i/stats.list" || exit 1
            find "$out_dir/iter$i/" -name "*.acc" -exec rm {} + || exit 1

            date > "$out_dir/iter$i/.done"
        else
            echo "Iteration $i has already been done. Skipping."
        fi
    done

    # Create a link pointing to the model of the last iteration.
    #ln -fs "$out_dir/iter$niter/model.bin" "$out_dir/model.bin"
    cp  "$out_dir/iter$niter/model.bin" "$out_dir/model.bin"

    date > "$out_dir/.done"
else
    echo "The model is already trained. Skipping."
fi

