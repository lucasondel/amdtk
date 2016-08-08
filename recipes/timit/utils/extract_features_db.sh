#!/usr/bin/env bash

#
# Extract features for the whole database.
#

if [ $# -ne 1 ]; then
    echo "usage: $0 setup.sh"
    exit 1
fi

setup="$1"
source "$setup" || exit 1

if [ ! -e "$fea_dir/.done" ]; then

    # Create the output directory.
    mkdir -p "$fea_dir"

    # Extract the features
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$fea_parallel_opts" \
        "extract-features" \
        "$scp" \
        "$PWD/utils/extract_features.sh $setup \$ITEM1 \$ITEM2" \
        "$fea_dir" || exit 1

    echo "The features have been extracted to $fea_dir."

    date > "$fea_dir"/.done
else
    echo The features have already been extracted. Skipping.
fi

