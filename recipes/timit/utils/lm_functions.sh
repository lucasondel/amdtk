
function sample_paths {
    local latt_dir="$1"
    local out_dir="$2"

    # We first label the training data by sampling from the lattices.
    if [ ! -e "$out_dir/.done" ]; then

        # Create the output directory.
        mkdir -p "$out_dir"

        # Sampling a path from the lattices.
        utils/phone_loop_sample_path.sh $setup $ac_weight $lm_weight \
            "$latt_dir" "$out_dir" || exit 1

        # Create a text file from the HTK label files.
        local labels_file=$(find "$out_dir" -name "*.lab") || exit 1

        for f in $labels_file ; do
            cat "$f" | awk '{printf $3 " "} END {print ""}' \
                >> "$out_dir/text" || exit 1
        done

        date > "$out_dir/.done"
    else
        echo "The sampling has already been done. Skipping."
    fi
}

function lm_create {
    local latt_dir="$1"
    local out_dir="$2"

    if [ ! -e "$out_dir/.done" ]; then
        # Label the data.
        sample_paths "$latt_dir" "$out_dir/sampled_paths"

        # Generate the vocabulary.
        for i in $(seq $truncation); do
            echo "a$i"
        done > "$out_dir/vocabulary.txt"

        # Create the LM model
        amdtk_lm_create --niter=0 --resample "$lm_params" \
            "$out_dir/vocabulary.txt" "$out_dir/sampled_paths/text" \
            "$out_dir/lm.bin" || exit 1

        date > "$out_dir/.done"
    else
        echo "The language model has already been created. Skipping."
    fi
}

function lm_train {
    local lm_dir="$1"
    local lm_model="$lm_dir/lm.bin"
    local latt_dir="$2"
    local out_dir="$3"

    if [ ! -e "$out_dir/.done" ]; then
        # Label the data.
        sample_paths "$latt_dir" "$out_dir/sampled_paths"

        # Create the LM model
        amdtk_lm_resample --resample "$lm_model" "$lm_dir/sampled_paths/text" \
            "$out_dir/sampled_paths/text" "$out_dir"/lm.bin || exit 1

        date > "$out_dir/.done"
    else
        echo "LM training iteration has already been done. Skipping."
    fi
}

