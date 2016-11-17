#!/usr/bin/env bash

set -e

# 
# Prepare the Xitsonga database. 
#

if [ $# -ne 1 ] 
then
    echo usage: $0 "<setup.sh>"
    exit 1
fi

source $1


if [ ! -e ${root}/data/.done ] 
then
    # create data directory
    mkdir -p ${root}/data
    
    # copy data for evaluation
    cp ${xitsonga_eval2}/resources/resources/xitsonga.{classes,phn,split,wrd} \
       ${root}/data
    
    # xitsonga va data set.
    find $xitsonga_wavs -type f -name *.wav | \
        python local/make_path.py data/xitsonga.split > data/va.scp
#     sed -i 's/$/_timed/g' data/va.scp
    cat data/va.scp | awk 'BEGIN {FS=":"} {print $2}' \
        > data/va.keys
  
    # xitsonga sil + va data set.
    find $xitsonga_wavs -type f -name *.wav | \
        python local/make_path.py > data/sil_va.scp
    cat data/sil_va.scp | awk 'BEGIN {FS=":"} {print $2}' \
        > data/sil_va.keys

#     # combine all datsets
#     cat data/va.scp data/sil_va.scp > data/all.scp
#     cat data/va.keys data/sil_va.keys > data/all.keys
    
    # convert transcriptions
    convert_ctm_to_mlf --zerospeech < data/xitsonga.phn > data/score.ref
    cut -d ' ' -f 1 data/xitsonga.phn | uniq > data/eval.keys
    
    echo $(date) > ${root}/data/.done
else
    echo The data is already prepared. Skipping.
fi
