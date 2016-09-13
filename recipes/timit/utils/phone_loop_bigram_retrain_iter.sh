#!/usr/bin/env bash

if [ $# -ne 5 ]; then
    echo "usage: $0 <setup.sh> <parallel_opts> <keys> <model_dir> <out_dir> "
    echo "                                                                  "
    echo "Single iteration for retraiing the phone-loop model with a bigram "
    echo "language model.                                                   "
    echo "                                                                  "
    exit 0
fi

setup="$1"
parallel_opts="$2"
keys="$3"
model_dir="$4"
out_dir="$5"

source $setup || exit 1

if [ ! -e "$out_dir/.done" ]; then
    mkdir -p "$out_dir"
    
    # Generate the vocabulary file.
    vocab_file="$out_dir/vocabulary.txt"
    for i in $(seq $truncation); do
        echo "a$i"
    done > "$vocab_file"
    echo "sil" >> "$vocab_file"

    # Label the training data to estimate the LM.
    $PWD/utils/phone_loop_label.sh \
        "$setup" \
        "$parallel_opts" \
        "$keys" \
        "$model_dir" \
        "$out_dir/labels" || exit 1 

    # Train the language model.
    amdtk_lm_create \
        --resample_hparams \
        --niter=5 \
        "$lm_params" \
        "$vocab_file" \
        "$out_dir/labels/text" \
        "$out_dir/lm.bin" || exit 1

    # Set the bigram LM to the phone-loop model.
    mkdir -p $out_dir/initial_model
    amdtk_ploop_set_lm \
        "$model_dir/model.bin" \
        "$out_dir/lm.bin" \
        "$out_dir/initial_model/model.bin" || exit 1

    # Retrain the model.
    utils/phone_loop_vb_iter.sh \
        "$setup" \
        "$parallel_opts" \
        "$keys" \
        "$bigram_ac_weight" \
        "$out_dir/initial_model" \
        "$out_dir/retrained_model" || exit 1
        
    # Take the model from the last iteration.
    cp "$out_dir/retrained_model/model.bin" "$out_dir/model.bin" || exit 1

    date > "$out_dir/.done"
else
    echo "The model is already re-trained. Skipping."
fi

