#!/usr/bin/env bash

# 
# Language model training and rescoring of AUD-generated lattices.
#

if [ $# -ne 1 ] 
    then
    echo "usage: $0 <setup.sh>"
    exit 1
fi

setup=$1

source $setup || exit 1

# Copy setup.sh to the experiment directory so that it can be re-run.
#if [ -e $root/$lm_model_type/setup.sh ]; then
#  diff $setup $root/$lm_model_type/setup.sh >/dev/null \
#    || echo "Warning: $lm_model_type/setup.sh differs; not overwriting" >&2
#else
#  mkdir -p $root/$lm_model_type
#  cp $setup $root/$lm_model_type/setup.sh
#fi

echo "word segmentation..."
utils/word_segmentation_run.sh $setup || exit 1
echo done


echo "phoneme language model estimation..."
utils/phone_lm_create.sh $setup || exit 1
echo done


echo "lattice rescoring..."
utils/rescore_lattices_db.sh $setup  $lattices_dir $rescoring_lattice_path || exit 1
echo done

