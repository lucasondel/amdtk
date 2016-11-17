#!/usr/bin/env bash

set -e

#
# score feature vectors with eval1 task
#

if [ $# -lt 3 -o $# -gt 4 ]; then
    echo "usage: $0 <setup.sh> <parallel_opts> <out_dir> [<segments_filename>]" 
    exit 1
fi

setup="$1"
source "$setup" || exit 1

parallel_opts="$2"
out_dir="$3"
segments_filename="$4"

if [ ! -e "$out_dir"/.done ]; then

  # run scoring
  amdtk_run $parallel_profile \
      --ntasks "$parallel_n_core" \
      --options "$parallel_opts" \
      "score-eval2" \
      "${out_dir}/score.list" \
      "$PWD/utils/score_eval2.sh $setup \$ITEM1 $out_dir $segments_filename" \
      "$out_dir" || exit 1

    date > "$out_dir"/.done
else
  echo "The segmentation result has been scored with eval2."
fi
