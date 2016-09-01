#!/usr/bin/env bash

# 
# Prepare the WSJ database. 
#

######################################################################
# Functions                                                          #
######################################################################

#
# Returns the last column from the input
#
function last_column { rev | cut -d ' ' -f 1 | rev; }

#
# Extracts the unique key from the input data
#
function get_key { awk 'BEGIN {FS=":"} {print $2}'; }

#
# Lists all regular files.
# This will not include current and the parent directory
#
function list_files { ls -la $1 | grep ^-; }

######################################################################

if [ $# -ne 1 ] 
    then
    echo usage: $0 "<setup.sh>"
    exit 1
fi

source $1 || exit 1


if [ ! -e ${root}/data/.done ] 
    then
    
    mkdir -p ${root}/data

    zc0_train="$zc0/train/wav"
    zc0_test="$zc0/devel_test/wav"

    # Training set
    list_files $zc0_train | last_column | python local/make_path.py $zc0_train > data/train.scp
    cat data/train.scp | get_key > data/train_unique.keys

    # Test devel set
    list_files $zc0_test | last_column | python local/make_path.py $zc0_test > data/devel_test.scp
    cat data/devel_test.scp | get_key > data/devel_test_unique.keys

    # We need to get a all the wav files in a single file for the features.
    cat data/*.scp | sort | uniq > data/all.scp
    cat data/all.scp | get_key > data/all.keys

    echo $(date) > ${root}/data/.done
else
    echo The $db data is already prepared. Skipping.
fi

