#!/usr/bin/env bash

counts=0
while [[ $# > 3 ]]
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

if [ $# -ne 3 ]; then
    echo 'usage:' $0 ' [--counts] <setup.sh> </path/to/posterior_file.htk> <out_dir>'
    exit
fi

source $1 || exit 1
pos_file=$2
out_dir=$3

base1=`basename ${pos_file}`
base_name=${base1%.*}

tmp_dir=`mktemp -d`
mkdir -p ${out_dir}
    
HVite -T 1 -y 'lab' -z 'latt' -l ${tmp_dir} \
    -C ${conf_latt_dir}/HVite.cfg   \
    -w ${conf_latt_dir}/monophones_lnet.hvite \
    -n 2 1 \
    -p ${penalty} \
    -q 'Atval' \
    -s ${gscale} \
    -t ${beam_thresh} \
    -i ${tmp_dir}/${base_name}.lab \
    -H ${conf_latt_dir}/hmmdefs.hvite \
    ${conf_latt_dir}/dict \
    ${conf_latt_dir}/phonemes \
    $pos_file || exit 1

if [ $counts -ne 0 ]; then	

    lattice-tool -in-lattice ${tmp_dir}/${base_name}.latt \
	-order ${latt_count_order} -compute-posteriors \
	-compact-expansion -read-htk -write-htk -write-ngrams \
	${out_dir}/${base_name}.counts || exit 1
fi

cp ${tmp_dir}/${base_name}.lab ${out_dir} || exit 1
gzip -c ${tmp_dir}/${base_name}.latt > ${out_dir}/${base_name}.latt.gz || exit 1

rm -r ${tmp_dir}



