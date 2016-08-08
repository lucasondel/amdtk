#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    exit 1
fi

setup=$1
source $setup || exit 1

if [ ! -e ${ws_base_path}/.done ]; then
    mkdir ${ws_base_path}

    latt_path=${ws_base_path}/WSJ_AMDTK_Lattice
    if [[ ! -e ${latt_path} ]]; then
        mkdir -p ${latt_path}
    fi

    # Extract the lattices with ending .latt into a local directory
    find "$lattice_base_path"/ -name "*.latt.gz" -print0 |
    while IFS= read -r -d '' htk_lattice_file; do
        out_lattice_file=${htk_lattice_file/'.latt.gz'/'.latt'}
        out_lattice_file=$latt_path/${out_lattice_file##*/}
        gunzip "-c" "$htk_lattice_file" > "$out_lattice_file"
    done

    # create filelist
    file_list=${latt_path}/wsj_amdtk_flist.txt
    ls $latt_path/*.latt > ${file_list}

    # run latticewordsegmentation
    utils/latticewordsegmentation.sh \
        "${file_list}" "${ws_output_path}" "${word_lm_order}" \
        "${char_lm_order}" "${ws_niter}" "${ws_AMScale}" "${ws_PruneFactor}" \
        2>&1 > "$ws_base_path/latticewordsegmentation.log" || exit 1

    # remove the copied and unpacked htk lattices (they  are not needed anymore)
    find "$latt_path" -name "*.latt" -print0 |
    while IFS= read -r -d '' lattice_file; do
        rm $lattice_file
    done

    # extract label files from timing file
    if [[ $output_wordsegmentation_labels ]]; then
        mkdir -p "$ws_label_dir"
        convert_segmentations.sh "$ws_output_timing" "$ws_label_dir" || true
    fi

    date > "$ws_base_path/.done"
else
    echo "Directory with wordsegmentation output exists. Skipping"
fi

