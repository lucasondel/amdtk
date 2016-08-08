#!/usr/bin/env bash

#
# Train the infinite phone-loop model.
#

if [ $# -ne 5 ]; then
    echo "usage: $0 <setup.sh> <niter> <lm_niter> <model_dir> <out_dir>"
    exit 1
fi

setup="$1"
niter="$2"
lm_niter="$3"
init_model_dir="$4"
out_dir="$5"

# Load utility functions.
source utils/lm_functions.sh
source utils/phone_loop_functions.sh

source $setup || exit 1

function retrain {
    local lm_niter="$1"
    local init_model_dir="$2"
    local init_lm_dir="$2/lm"
    local keys="$3"
    local out_dir="$4"

    if [ ! -e "$out_dir/.done" ]; then
        mkdir -p "$out_dir"

        echo generating lattices...
        generate_lattices "$init_model_dir" "$keys" "$out_dir/lattices" 
        
        echo creating the bigram LM...
        lm_create "$out_dir/lattices" "$out_dir/lm0"

        echo training the bigram LM...
        for i in $(seq $lm_niter); do
            lm_train "$out_dir/lm$((i-1))" "$out_dir/lattices" "$out_dir/lm$i"
        done

        echo Updating the phone loop model...
        utils/phone_loop_add_lm.sh $setup "$init_model_dir" \
            "$out_dir/lm$lm_niter" "$out_dir/init_model" || exit 1

        echo retraining the phone loop model...
        utils/phone_loop_train.sh $setup 5 "$out_dir/init_model" \
        "$out_dir" || exit 1

        date > "$out_dir/.done"
    else
        echo Re-training iteration is already done. Skipping.
    fi
}

if [ ! -e "$out_dir/.done" ]; then
    mkdir -p "$out_dir"

    # Initialize the retraining
    mkdir -p "$out_dir/iter0"
    cp "$init_model_dir/model.bin" "$out_dir/iter0/model.bin"

    # Retraining.
    for i in $(seq $niter); do
        echo "------------------------"
        echo "re-training iteration: $i"
        echo "------------------------"
        retrain "$lm_niter" "$out_dir/iter$((i-1))" "$train_keys" \
            "$out_dir/iter$i" || exit 1
    done
        
    # Take the model from the last iteration.
    cp "$out_dir/iter$niter/model.bin" "$out_dir"

    date > "$out_dir/.done"
else
    echo "The model is already re-trained. Skipping."
fi

