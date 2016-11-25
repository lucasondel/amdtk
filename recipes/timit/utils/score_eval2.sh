#!/usr/bin/env bash

set -e

#
# score the features in <features_dir> with eval1
#

if [ $# -lt 3 -o $# -gt 4 ] 
    then
    echo usage: $0 "<setup.sh> <seg_result> <out_dir> [<segments_filename>]"
    exit 1
fi

setup=$1
# Load the configuration.
source ${setup}

seg_result=$2
out_dir=$3
if [ "$4" != "" ]; then
  options="--segments_file $4"
fi

score_dir=${out_dir}/$(echo ${seg_result}|sed "s#${root}/${model_type}/##")

# scoring
if [ ! -e "${score_dir}/.done" ]
then
  mkdir -p ${score_dir}
  convert_ctm_to_eval2_cluster $options ${seg_result} ${score_dir}/classes
  ${eval2_tool} ${score_dir}/classes ${score_dir}
  date > ${score_dir}/.done
else
  echo "Scoring has already been done. Skipping."
fi
