#!/usr/bin/env bash

if [ $# -ne 5 ]; then
    echo ""
    echo "Label a data set with the infinite phone-loop model."
    echo ""
    echo "usage: $0 <setup.sh> <parallel_opts>  <keys> <model_dir> <label_dir>"
    exit 1
fi

setup="$1"
parallel_opts="$2"
keys="$3"
model="$4/model.bin"
label_dir="$5"

source "$setup" || exit 1

if [ ! -e "$label_dir/.done" ]; then

    # Create the output directory.
    mkdir -p "$label_dir"

    # Labeling
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$parallel_opts" \
        "pl-label" \
        "$keys" \
        "amdtk_ploop_label $model $fea_dir/\${ITEM1}.${fea_ext} \
        $label_dir/\${ITEM1}.lab" \
        "$label_dir" || exit 1
    
    # Create a text file from the HTK labels. This text file is needed 
    # for training the language model.
    labels_file=$(find "$label_dir" -name "*.lab") || exit 1
    for f in $labels_file ; do
        cat "$f" | awk '{printf $3 " "} END {print ""}' >> "$label_dir/text" \
            || exit 1
    done

    date > "$label_dir/.done"
else
    echo "The labeling has already been done. Skipping."
fi

