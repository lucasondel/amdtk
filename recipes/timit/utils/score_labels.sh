#!/usr/bin/env bash

#
# Performs scoring based on labels from AUD clusters
#


if [ $# -ne 3 ]; then
    echo usage: $0 "<setup.sh> <label_dir> <out_dir>"
    exit 1
fi

setup=$1
label_dir=$2
out_dir=$3

score_lbs="$out_dir/score.lab"
score_res="$out_dir/scores"

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
    lab_files=$(awk -v label_dir="$label_dir" '{ print label_dir "/" $0 ".lab"}' $score_keys)
    amdtk_concat --htk $lab_files $score_lbs
fi

# perform scoring if not done already
if [ ! -e $score_res ]; then
    amdtk_score_labels $score_ref $score_lbs | tee $score_res
else
    echo "Scoring already performed. Scores:"
    cat $score_res
fi


