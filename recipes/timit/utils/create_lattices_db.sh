#!/usr/bin/env bash

#
# Generate the lattices for the whole database.
#

if [ $# -ne 5 ]; then
    echo "usage: $0 <setup.sh> <parallel_opts> <keys> <model_dir> <out_dir> "
    echo "                                                                  "
    echo "Generate the lattices for the whole database.                     "
    echo "                                                                  "
    exit 0
fi

setup="$1"
parallel_opts="$2"
keys="$3"
model="$4/model.bin"
out_dir="$5"

source "$setup" || exit 1

utils/create_htk_lattice_config.sh "$setup" || exit 1

if [ ! -e "$out_dir/.done" ]; then

    # Create the output _directory.
    mkdir -p "$out_dir"

    
    # Generating the posteriors. Here we use the "HTK trick" as we want
    # to build HTK lattices from the posteriors.
    mkdir -p "$out_dir/posts"
    amdtk_run \
        $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$parallel_opts" \
        --python_script \
        "pl-post"  \
        "$keys" \
        "amdtk_ploop_post --htk_trick --hmm_states \
            $model $fea_dir/\${ITEM1}.${fea_ext} \
            $out_dir/posts/\${ITEM1}.pos" \
        "$out_dir/posts" || exit 1

    find "$out_dir/posts" -name "*.pos" > "$out_dir"/posts.list

    # Generating lattices.
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$parallel_opts" \
        "gen-latt" \
        "$out_dir/posts.list" \
        "utils/create_lattice.sh $setup \$ITEM1 $out_dir" \
        "$out_dir" || exit 1

    date > "$out_dir/.done"
else
    echo "The lattices have already been generated. Skipping."
fi

