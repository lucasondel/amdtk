#!/usr/bin/env bash

counts=0
while [[ $# > 2 ]]
do
    key="$1"
    
    case $key in
	--counts)
	    counts=1
	    ;;
	*)
            # unknown option
	    ;;
    esac
    shift # past argument or value
done

if [ $# -ne 2 ]; then
    echo 'usage:' $0 ' [--counts] <setup.sh> </path/to/posterior_file.htk>'
    exit
fi

source $1 || exit 1
pos_file=$2

base1=`basename ${pos_file}`
base_name=${base1%.*}

mkdir -p ${out_nbest_dir}
    
HVite -T 1 -l ${out_nbest_dir} \
    -C ${conf_latt_dir}/HVite.cfg   \
    -w ${conf_latt_dir}/monophones_lnet.hvite \
    -n $nt $N \
    -p ${penalty} \
    -q 'Atval' \
    -s ${gscale} \
    -t ${beam_thresh} \
    -i ${out_nbest_dir}/${base_name}.lab \
    -H ${conf_latt_dir}/hmmdefs.hvite \
    ${conf_latt_dir}/dict \
    ${conf_latt_dir}/phonemes \
    $pos_file || exit 1




