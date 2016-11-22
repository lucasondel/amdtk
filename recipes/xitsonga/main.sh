#!/usr/bin/env bash

set -e

# Run first pass of amdtk with mfcc features
# Note: adjust settings in setup.sh before running
./run.sh $(pwd -P)/setup_mfcc.sh

# Run kaldi recipe to extract the transformed features
# Note: adjust settings in path.sh and setup.sh before running
kaldi_recipe_dir='/scratch/owb/Downloads/amdtk_feature_transform/egs/xitsonga'
pushd "$kaldi_recipe_dir"
./run.sh
ln -s "$kaldi_recipe_dir/data/train_fmllr_lda" plp_fmllr_lda
popd

# Run second pass of amdtk with transformed features
./run.sh $(pwd -P)/setup_fmllr.sh
