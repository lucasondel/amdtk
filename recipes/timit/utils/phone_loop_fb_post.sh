#!/usr/bin/env bash

hmm_states=""
htk_trick=""
segments_file=""
as_text=""
while [[ $# > 5 ]]
do
    key="$1"
    
    case $key in
	--hmm_states)
	    hmm_states="--hmm_states"
	    ;;
        --htk_trick)
            htk_trick="--htk_trick"
            ;;
        --as_text)
            as_text="--as_text"
            ;;
        --segments_file)
            segments_file="--segments_file $2"
            shift
            ;;
	*)
        # unknown option
	    ;;
    esac
    shift # past argument or value
done

if [ $# -ne 5 ]; then
    echo $@
    echo "usage: $0 [--hmm_states|--as_text|--htk_trick|--segments_file segments file] <setup.sh> <parallel_opts> <keys> <model_dir> <out_dir> "
    echo "                                                                                "
    echo "Generate the unit or state per frame posteriors.                                         "
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
        --python_script \
        "pl-post"  \
        "$keys" \
        "amdtk_ploop_fb_post --ac_weight $post_ac_weight $segments_file $htk_trick $hmm_states $as_text $model \
            $fea_dir/\${ITEM1}.${fea_ext} $out_dir/\${ITEM1}.pos" \
        "$out_dir" || exit 1

    date > "$out_dir/.done"
else
    echo "The posteriors have already been generated. Skipping."
fi

