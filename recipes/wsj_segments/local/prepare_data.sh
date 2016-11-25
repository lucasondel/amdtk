#!/usr/bin/env bash

# 
# Prepare the WSJ database. 
#

if [ $# -ne 1 ] 
    then
    echo usage: $0 "<setup.sh>"
    exit 1
fi

source $1 || exit 1


if [ ! -e ${root}/data/.done ] 
    then
    
    mkdir -p ${root}/data 

    # convert kaldi alignments to mlf for NMI evaluation
    kaldi_ali_to_mlf \
        --split_file data/wsj.split \
        --non_speech_phones "SIL NSN SPN" \
        data/training_unique.phonealignment_wordpos.txt \
        data/score.ref

    # SI 84 training set.
    cat $wsj0/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
        grep -v ';' | grep -v -i "11_2_1:wsj0/si_tr_s/401"| \
        python local/make_path.py --split_file data/wsj.split \
            $wsj0 > data/training_si84.scp
    cat data/training_si84.scp | awk 'BEGIN {FS=":"} {print $2}' \
        > data/training_si84.keys
    
    # SI 284 training set.
    cat $wsj1/13-34.1/wsj1/doc/indices/si_tr_s.ndx | grep -v ';' | \
        python local/make_path.py --split_file data/wsj.split \
            $wsj1 > data/training_si284.scp
    cat data/training_si84.scp >> data/training_si284.scp
    cat data/training_si284.scp | awk 'BEGIN {FS=":"} {print $2}' \
        > data/training_si284.keys

    # November 92 test set.
    cat $wsj0/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
        grep -v ';' | grep -v -i "11_2_1:wsj0/si_tr_s/401"| \
        python local/make_path.py --split_file data/wsj.split \
            --add_ext $wsj0 > data/test_eval92.scp
    cat data/test_eval92.scp | awk 'BEGIN {FS=":"} {print $2}' \
        > data/test_eval92.keys

    # We need to get a all the wav files in a single file for the features.
    cat data/*.scp | sort | uniq > data/all.scp
    cat data/all.scp | awk 'BEGIN {FS=":"} {print $2}' > data/all.keys

#     python local/get_docs_from_prompts.py $wsj0 $wsj1
# 
#     cat data/training_unique.keys data/test_unique.keys > data/all_unique.keys
    
    # Get all wav files for which we have alignments to evaluate NMI
    cut -d ' ' -f 1 data/training_unique.phonealignment_wordpos.txt \
        > data/training_unique.keys
    grep -f data/training_unique.keys data/all.scp > data/training_unique.scp
    
    echo $(date) > ${root}/data/.done
else
    echo The $db data is already prepared. Skipping.
fi

