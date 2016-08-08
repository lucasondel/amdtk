#!/usr/bin/env bash

if [ $# -ne 1 ]
    then
    echo usage: $0 "<setup.sh>"
    exit 1
fi

source $1 || exit 1

if [ ! -e ${conf_latt_dir}/.done ] ; then

    mkdir -p ${conf_latt_dir}

    echo "TARGETKIND     = USER" > ${conf_latt_dir}/HVite.cfg

    # Create a pseudo phonemes list. The number of phonemes is define by the
    # truncation of the phone loop model.
    for p in `seq $truncation`; do
	echo 'a'${p} >> ${conf_latt_dir}/phonemes
	for s in `seq $nstates`; do
	    echo 'a'${p}__${s} >>  ${conf_latt_dir}/states
	done
    done
    cp ${conf_latt_dir}/phonemes ${conf_latt_dir}/hmmlist
    cat ${conf_latt_dir}/hmmlist | awk '{print $1,$1}' > ${conf_latt_dir}/dict

    # Create recognition net.
    HBuild ${conf_latt_dir}/hmmlist ${conf_latt_dir}/monophones_lnet.hvite

    # Create the HTK HMM definitions file.
    utils/create_HMM_def.sh ${conf_latt_dir}/states ${conf_latt_dir}/hmmdefs.hvite

    date > ${conf_latt_dir}/.done
fi
