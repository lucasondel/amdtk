#!/bin/bash

set -e

if [ $# -ne 4 ] 
    then
    echo "usage: $0 <setup.sh> <parallel_opts> <lattice_base_path> <out_dir>"
    exit 1
fi

setup="$1"
source ${setup}

parallel_opts="$2"
lattice_base_path="$3"
out_dir="$4"

mkdir -p ${out_dir}

# create params list
rm -f ${out_dir}/params.list
if [ ! -e "${out_dir}/params.list" ]; then
  for wlm in ${word_lm_order}
  do
    for clm in ${char_lm_order}
    do
      for addlm in ${addchar_lm_order}
      do
        for amscale in ${ws_AMScale}
        do
          for pruning in ${ws_PruneFactor}
          do
            echo "${wlm}/${clm}/${addlm}/${amscale}/${pruning}" >> ${out_dir}/params.list
          done
        done
      done
    done
  done
else
  echo "Parameter list already created. Skipping."
fi

latt_path=${out_dir}/lattices

mkdir -p ${latt_path}
file_list=${latt_path}/lattice_flist.txt

# Extract the lattices with ending .latt into a local directory and create file list
if [ ! -e "${file_list}" ]
then
  find "$lattice_base_path"/ -name "*.latt.gz" -print0 |
  while IFS= read -r -d '' htk_lattice_file
  do
      out_lattice_file=${htk_lattice_file/'.latt.gz'/'.latt'}
      out_lattice_file=$latt_path/${out_lattice_file##*/}
      gunzip "-c" "$htk_lattice_file" > "$out_lattice_file"
  done
  ls $latt_path/*.latt > ${file_list}
else
  echo "Lattice file list already exists. Skipping extraction!"
fi

if [ ! -e "$out_dir"/.done ]; then

  # run word segmentation
  amdtk_run $parallel_profile \
      --ntasks "$parallel_n_core" \
      --options "$parallel_opts" \
      "latticewordseg" \
      "${out_dir}/params.list" \
      "$PWD/utils/latticewordsegmentation.sh ${setup} ${file_list} ${out_dir} ${ws_niter} \$ITEM1" \
      "$out_dir" || exit 1

      #Item1: KnownN/UnkN/AddCharN/AmScale/PruneFactor
      
    date > "$out_dir"/.done
else
  echo "Segmentation has already been run. Skipping."
fi
