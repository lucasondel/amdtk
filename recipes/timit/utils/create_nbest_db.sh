#!/usr/bin/env bash

#
# Generate the lattices for the whole database.
#

if [ $# -ne 1 ]; then
    echo "usage: $0 <setup.sh>"
    exit 1
fi

setup=$1
source "$setup" || exit 1

# utils/create_htk_lattice_config.sh "$setup" || exit 1

if [ ! -e "$out_nbest_dir"/.done ]; then

    # Create the output _directory.
    mkdir -p "$out_nbest_dir"
    find "$post_dir" -name "*.htk" > "$out_nbest_dir"/posts.list

    # N-best-ing <- Santosh wrote that !
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$latt_parallel_opts" \
        "gen-nbest" \
        "$out_nbest_dir/posts.list" \
        "utils/create_nbest.sh $setup \$ITEM1" \
        "$out_nbest_dir" || exit 1

    date > "$out_nbest_dir"/.done
else
    echo "The n-best sequences have already been generated. Skipping."
fi

