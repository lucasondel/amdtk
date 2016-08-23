#!/usr/bin/env bash

if [ $# -ne 2 ]; then
    echo "usage: $0 <setup.sh> <parallel_opts>           "
    echo "                                               "
    echo "Extract the features for the whole database.   "
    echo "                                               "
    exit 1
fi

setup="$1"
parallel_opts="$2"
out_dir="$3"

source "$setup" || exit 1

if [ ! -e "$fea_dir/.done" ]; then

    # Create the output directory.
    mkdir -p "$fea_dir"

    # Extract the features
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$parallel_opts" \
        "extract-features" \
        "$scp" \
        "$PWD/utils/extract_features.sh $setup \$ITEM1 \$ITEM2" \
        "$fea_dir" || exit 1

    date > "$fea_dir"/.done
else
    echo The features have already been extracted. Skipping.
fi

