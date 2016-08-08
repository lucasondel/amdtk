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

    # SI 84 training set. - Contains pronounced punctuations
    # cat $wsj0/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
    #    grep -v ';' | grep -v -i "11_2_1:wsj0/si_tr_s/401"| \
    #    python local/make_path.py $wsj0 > data/training_si84.scp
    #cat data/training_si84.scp | awk 'BEGIN {FS=":"} {print $2}' \
    #    > data/training_si84.keys
    
    # SI 284 training set.
    cat $wsj1/13-34.1/wsj1/doc/indices/si_tr_s.ndx | grep -v ';' | \
        python local/make_path.py $wsj1 > data/training_si284.scp
    # cat data/training_si84.scp >> data/training_si284.scp
    cat data/training_si284.scp | awk 'BEGIN {FS=":"} {print $2}' \
        > data/training_si284.keys

    # November 92 test set. - Contains pronounced punctuations
    #cat $wsj0/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
    #    grep -v ';' | grep -v -i "11_2_1:wsj0/si_tr_s/401"| \
    #    python local/make_path.py --add_ext $wsj0 > data/test_eval92.scp
    #cat $wsj0/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
    #    grep -v ';' | grep -v -i "11_2_1:wsj0/si_tr_s/401"| \
    #    python local/make_path.py --add_ext $wsj0 > data/test_eval92.scp

    # Nov'93: (213 utts)
    # Have to replace a wrong disk-id.
    cat $wsj1/13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx | \
        sed s/13_32_1/13_33_1/ | grep -v ';' | \
        python local/make_path.py $wsj1 > data/test_eval93.scp

    # Nov'93: (213 utts, 5k)
    cat $wsj1/13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx | \
        sed s/13_32_1/13_33_1/ | grep -v ';' | \
        python local/make_path.py $wsj1 > data/test_eval93_5k.scp

    # Dev-set for Nov'93 (503 utts)  
    cat $wsj1/13-34.1/wsj1/doc/indices/h1_p0.ndx | grep -v ";" | \
        grep -v -i "13_16_1:wsj1/si_dt_20/4ka/" | \
        python local/make_path.py $wsj1 > data/test_dev93.scp

    # Dev-set for Nov'93 (513 utts, 5k vocab)  
    cat $wsj1/13-34.1/wsj1/doc/indices/h2_p0.ndx | grep -v ";" | \
        grep -v -i "13_16_1:wsj1/si_dt_20/4ka/" | \
        python local/make_path.py $wsj1 > data/test_dev93_5k.scp

    cat data/test_*.scp | sort | uniq > data/test_93.scp
    cat data/test_93.scp | awk 'BEGIN {FS=":"} {print $2}' \
        > data/test_93.keys

    # We need to get a all the wav files in a single file for the features.
    cat data/*.scp | sort | uniq > data/all.scp
    cat data/all.scp | awk 'BEGIN {FS=":"} {print $2}' > data/all.keys

    python local/get_docs_from_prompts.py $wsj1

    cat data/training_unique.keys data/test_unique.keys > data/all_unique.keys

    echo $(date) > ${root}/data/.done
else
    echo The $db data is already prepared. Skipping.
fi

