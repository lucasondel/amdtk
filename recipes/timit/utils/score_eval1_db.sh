#!/usr/bin/env bash

#
# score feature vectors with eval1 task
#

if [ $# -ne 2 ]; then
    echo "usage: $0 setup.sh <out_dir>" 
    exit 1
fi

setup="$1"
source "$setup" || exit 1

out_dir="$2"

if [ ! -e "$out_dir"/.done ]; then

  # run scoring
  amdtk_run $parallel_profile \
      --ntasks "$parallel_n_core" \
      --options "$eval1_parallel_opts" \
      "score-eval1" \
      "${out_dir}/list" \
      "$PWD/utils/score_eval1.sh $setup \$ITEM1 $out_dir" \
      "$out_dir" || exit 1

    date > "$out_dir"/.done
else
  echo "The feature vectors have been scored with eval1."
fi
