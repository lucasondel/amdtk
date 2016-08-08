#!/usr/bin/env bash

#
# Label a data set with the infinite phone-loop model.
#

if [ $# -ne 3 ]; then
    echo "usage: $0 <setup.sh> <model_dir> <label_dir>"
    exit 1
fi

setup=$1
model="$2/model.bin"
label_dir="$3"

source "$setup" || exit 1

if [ ! -e "$label_dir"/.done ]; then

    # Create the output directory.
    mkdir -p "$label_dir"

    # Labeling
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$label_parallel_opts" \
        "pl-label" \
        "$label_keys" \
        "amdtk_ploop_label $model $fea_dir/\${ITEM1}.${fea_ext} \
        $label_dir/\${ITEM1}.lab" \
        "$label_dir" || exit 1
    
    # Create a text file from the HTK labels.
    labels_file=$(find "$label_dir" -name "*.lab") || exit 1
    for f in $labels_file ; do
        cat "$f" | awk '{printf $3 " "} END {print ""}' >> "$label_dir"/text \
            || exit 1
    done

    date > "$label_dir"/.done
else
    echo "The labeling has already been done. Skipping."
fi

