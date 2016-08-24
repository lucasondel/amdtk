#!/usr/bin/env bash

hmm_states=""
while [[ $# > 5 ]]
do
    key="$1"
    
    case $key in
	--hmm_states)
	    hmm_states="--hmm_states"
	    ;;
	*)
        # unknown option
	    ;;
    esac
    shift # past argument or value
done

if [ $# -ne 5 ]; then
    echo "usage: $0 [--hmm_states] <setup.sh> <parallel_opts <keys> <model_dir> <out_dir> "
    echo "                                                                                "
    echo "Generate the unit per frame posteriors.                                         "
    echo "                                                                                "
    exit 1
fi

setup="$1"
parallel_opts="$2"
keys="$3"
model="$4/model.bin"
out_dir="$5"

source $setup || exit 1

if [ ! -e "$out_dir/.done" ]; then

    # Create the output _directory.
    mkdir -p "$out_dir"

    # Generating the posteriors. Here we use the "HTK trick" as we want
    # to build HTK lattices from the posteriors.
    amdtk_run $parallel_profile \
        --ntasks "$parallel_n_core" \
        --options "$parallel_opts" \
        "pl-post"  \
        "$keys" \
        "amdtk_ploop_fb_post --ac_weight $post_ac_weight $hmm_states $model \
            $fea_dir/\${ITEM1}.${fea_ext} $out_dir/\${ITEM1}.pos" \
        "$out_dir" || exit 1

    date > "$out_dir/.done"
else
    echo "The posteriors have already been generated. Skipping."
fi

