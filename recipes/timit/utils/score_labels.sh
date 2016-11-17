#!/usr/bin/env bash

#
# Performs scoring based on labels from AUD clusters
#


if [ $# -lt 4 -o $# -gt 7 ]; then
    echo usage: $0 "<setup.sh> <keys> <label_dir> <out_dir> [--htk|--timit|--mlf] [--segments_file segmentsfile]"
    exit 1
fi

setup="$1"
keys="$2"
label_dir="$3"
out_dir="$4"

shift 4
while [ $# -gt 0 ]
do
    case $1 in
        --htk|--timit|--mlf)
            label_format="$1"
            ;;
        --segments_file)
            segments_file="$1 $2"
            shift
            ;;
    esac
    shift
done

if [ "$label_format" = "" ]; then
    label_format="--htk"
fi

score_lbs="$out_dir/score.lab"
score_res="$out_dir/scores"
score_ref="data/score.ref"

# load the configuration
source $setup || exit 1

# create master label file for TIMIT if it doesn't exist (in the dir of run.sh)
if [ ! -e $score_ref ]; then
    echo "Concatenating master labels for TIMIT..."
    amdtk_concat --timit --add_dirname 1 "$db_path/test/*/*/*.phn" $score_ref
fi

mkdir -p $out_dir

# extract all labels from current run, if not done so already
if [ ! -e $score_lbs ]; then
    echo "Concatenating AUD labels..."
    lab_files=$(awk -v label_dir="$label_dir" '{ print label_dir "/" $0 ".lab"}' $keys)
    amdtk_concat $label_format $lab_files $score_lbs
fi

# perform scoring if not done already
if [ ! -e $score_res ]; then
    amdtk_score_labels $segments_file $score_ref $score_lbs | tee $score_res
else
    echo "Scoring already performed. Scores:"
    cat $score_res
fi


