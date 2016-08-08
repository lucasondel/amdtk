#!/usr/bin/env bash

#
# Generate the lattices for the whole database.
#

if [ $# -ne 3 ]; then
    echo "usage: $0 <setup.sh> <post_dir> <out_dir>"
    exit 1
fi

setup=$1
post_dir=$2
out_dir=$3

source "$setup" || exit 1

utils/create_htk_lattice_config.sh "$setup" || exit 1

if [ ! -e "$out_dir"/.done ]; then

    # Create the output _directory.
    mkdir -p "$out_dir"
    find "$post_dir" -name "*.htk" > "$out_dir"/posts.list

    # Latticing <- Santosh wrote that !
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$latt_parallel_opts" \
        "gen-latt" \
        "$out_dir/posts.list" \
        "utils/create_lattice.sh $setup \$ITEM1 $out_dir" \
        "$out_dir" || exit 1

    date > "$out_dir"/.done
else
    echo "The lattices have already been generated. Skipping."
fi

