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

block_size=10000
if [ ! -e "$fea_dir/.done" ]; then

    # Create the output directory.
    mkdir -p "$fea_dir"

    num_jobs=$(wc -l $scp | cut -d' ' -f1)
    current_file=1
    block=1
    while [ $current_file -lt $num_jobs ]
    do
        echo "$current_file < $num_jobs ?"
        # Get the end of t block
        end_of_block=$(($current_file + $block_size - 1))
        
        # The last file in the block is the min of adding a whole
        # block and going to the end of the .scp file
        last_file=$(($end_of_block<$num_jobs?$end_of_block:$num_jobs))
        
        # Extract all of the rows that I need
        sed -n ${current_file},${last_file}p $scp > ${scp}.${block}

        # Update iterators for next loop
        current_file=$(($last_file+1))
        # Extract the features
        amdtk_run $parallel_profile \
            --ntasks "$parallel_n_core" \
            --options "$fea_parallel_opts" \
            "extract-features" \
            "${scp}.${block}" \
            "$PWD/utils/extract_features.sh $setup \$ITEM1 \$ITEM2" \
            "$fea_dir" || exit 1
        
        block=$(($block+1))
    done
    
    echo "The features have been extracted to $fea_dir."
    date > "$fea_dir"/.done
    
else
    echo The features have already been extracted. Skipping.
fi

