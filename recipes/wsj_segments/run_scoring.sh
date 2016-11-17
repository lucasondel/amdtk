#!/usr/bin/env bash

# July 18, 2016, Santosh

if [ $# -ne 1 ]; then
    echo 'usage: '$0 'setup_eval.sh'
    exit
fi

setup=$1
path=$PWD/path.sh

source $setup
source $path

if [ ! -d ${exp_dir} ]; then
    mkdir -p ${exp_dir}
fi

echo -e "[queues]\ndefault="${default}"\ntemp="${temp}"\nlong="${long}\
"\n[paths]\ntmp_d="${tmp_d}"\npath_env="${path_env} > ${sge_cfg_f}

sh ${PWD}/../../scoring/cluster_docs.sh $setup
