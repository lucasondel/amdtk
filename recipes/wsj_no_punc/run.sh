#!/usr/bin/env bash

# 
# Acoustic Unit Discovery based on infinite phone-loop model.
#

if [ $# -ne 1 ] 
    then
    echo "usage: $0 <setup.sh>"
    exit 1
fi

setup=$1

source $setup || exit 1

# Copy setup.sh to the experiment directory so that it can be re-run.
if [ -e $root/$model_type/setup.sh ]; then
  diff $setup $root/$model_type/setup.sh >/dev/null \
    || echo "Warning: $model_type/setup.sh differs; not overwriting" >&2
else
  mkdir -p $root/$model_type
  cp $setup $root/$model_type/setup.sh
fi


n=0 

echo "($((++n))) Data preparation..."
local/prepare_data.sh $setup || exit 1
echo done

echo "($((++n))) Features extraction..."
utils/extract_features_db.sh $setup || exit 1
echo done

echo "($((++n))) Creating the model..."
utils/phone_loop_create.sh $setup $root/$model_type/initial_model || exit 1
echo done

echo "($((++n))) Training the model with unigram LM..."
utils/phone_loop_train.sh $setup 10 $root/$model_type/initial_model \
    $root/$model_type/unigram || exit 1
echo done

echo "($((++n))) Labeling the unigram model..."
utils/phone_loop_label.sh $setup  $root/$model_type/unigram \
    $root/$model_type/unigram_labels || exit 1
echo done

echo "($((++n))) Generating posteriors..."
utils/phone_loop_post.sh $setup $root/$model_type/unigram \
    $root/$model_type/unigram_posts || exit 1
echo done

echo "($((++n))) Generating lattices..."
utils/create_lattices_db.sh $setup $root/$model_type/unigram_posts \
    $root/$model_type/unigram_lattices || exit 1
echo done

