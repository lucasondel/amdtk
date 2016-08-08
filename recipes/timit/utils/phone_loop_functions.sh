
function generate_posteriors {
    local model="$1/model.bin"
    local keys="$2"
    local out_dir="$3"

    if [ ! -e "$out_dir"/.done ]; then

        # Create the output _directory.
        mkdir -p "$out_dir"

        # Generating the posteriors. Here we use the "HTK trick" as we want
        # to build HTK lattices from the posteriors.
        amdtk_run $parallel_profile \
            --ntasks "$parallel_n_core" \
            --options "$post_parallel_opts" \
            "pl-post"  \
            "$keys" \
            "amdtk_ploop_post --htk_trick --hmm_states $model \
                $fea_dir/\${ITEM1}.${fea_ext} \
            $out_dir/\${ITEM1}.htk" \
            "$out_dir" || exit 1

        date > "$out_dir"/.done
    else
        echo "The posteriors have already been generated. Skipping."
    fi
}

function generate_lattices {
    local model_dir="$1"
    local keys="$2"
    local out_dir="$3"

    # Generate the posteriors for the lattice generation.
    generate_posteriors "$model_dir" "$keys" "$out_dir/posts" || exit 1

    # Prepare HTK lattice generation. 
    utils/create_htk_lattice_config.sh "$setup" || exit 1

    if [ ! -e "$out_dir"/.done ]; then
        # Create the output _directory.
        mkdir -p "$out_dir"

        # Extract the list of posteriors.
        find "$out_dir/posts" -name "*.htk" > "$out_dir/posts.list"

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
}

