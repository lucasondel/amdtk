#!/usr/bin/env bash

# 
# Prepare the TIMIT database. We expect the database to be store as:
# /path/to/db/{train,test}/dr[1-8]/spk_id/sentence_id.wav
#

if [ $# -ne 1 ] 
    then
    echo usage: $0 "<setup.sh>"
    exit 1
fi

source $1 || exit 1



if [ ! -e ${root}/data/.done ] 
    then
    
    echo ${db} path : ${db_path}

    mkdir -p ${root}/data 

    # Training utterances. 
    for path in `find $db_path/TRAIN -path */*/*.WAV` ; 
    do
        filename=`basename $path`
        filename=${filename%.*}
        dir=`dirname $path`
        dir=`basename $dir`
        key=${dir}_${filename}

        echo $path:$key >> ${root}/data/all.scp
        echo $key >> ${root}/data/all.keys
        echo $key >> ${root}/data/train.keys
    done

    # Test utterances.
    for path in `find $db_path/TEST -path */*/*.WAV` ; 
    do
        filename=`basename $path`
        filename=${filename%.*}
        dir=`dirname $path`
        dir=`basename $dir`
        key=${dir}_${filename}

        echo $path:$key >> ${root}/data/all.scp
        echo $key >> ${root}/data/all.keys
        echo $key >> ${root}/data/test.keys
    done

    echo Data stored in ${root}/data.
    echo "# utterances for training: `cat ${root}/data/train.keys | wc -l`"
    echo "# utterances for testing: `cat ${root}/data/test.keys | wc -l`"

    echo `date` > ${root}/data/.done
else
    echo The $db data is already prepared. Skipping.
fi

