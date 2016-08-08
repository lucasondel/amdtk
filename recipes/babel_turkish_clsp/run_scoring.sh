#!/usr/bin/env bash

# July 20, 2016, Santosh

if [ $# -ne 1 ]; then
    echo 'usage: '$0 './setup_scorinig.sh'
    exit
fi

setup=${PWD}/$1
path=${PWD}/../timit/path.sh

source $setup
source $path

if [ ! -d ${exp_dir} ]; then
    mkdir -p ${exp_dir}
fi

echo -e "[queues]\ndefault="${default}"\ntemp="${temp}"\nlong="${long}\
"\n[paths]\ntmp_d="${tmp_d}"\npath_env="${path_env} > ${sge_cfg_f}

sh ${PWD}/../../scoring/cluster_docs.sh $setup
