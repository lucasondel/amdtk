#!/usr/bin/env bash

#
# score the features in <features_dir> with eval1
#

if [ $# -ne 3 ] 
    then
    echo usage: $0 "<setup.sh> <features_dir> <out_dir>"
    exit 1
fi

setup=$1
# Load the configuration.
source $setup || exit 1

features_dir=$2
out_dir=$3
scoring_dir=${out_dir}/$(echo ${features_dir}|sed "s#${root}/${model_type}/##")
scoring_features_dir=${scoring_dir}/features

# scoring
if [ ! -e "${scoring_dir}/.done" ]
then
  mkdir -p ${scoring_features_dir}
  find ${features_dir} -type f -name '*.pos' -exec ln -sf {} ${scoring_features_dir} \;
  ${eval1_tool} ${scoring_features_dir} ${scoring_dir}
  date > ${scoring_dir}/.done
else
  echo "The features have already been scored. Skipping."
fi
