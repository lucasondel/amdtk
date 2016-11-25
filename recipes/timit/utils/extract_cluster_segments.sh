#!/bin/bash

set -e

if [ $# -lt 4 -o $# -gt 13 ]; then
    echo "usage: $0 <setup.sh> <ctm_file> <wav_scp> <out_dir> [--segments_file segments file|--file_type file format|--mlf mlf|--sil_labels sil labels (comma seperated list)|--remove_stress]"
    echo "                                               "
    echo "Extract clustered segments for the segmentation result."
    echo "                                               "
    exit 1
fi

setup="$1"
ctm_file="$2"
wav_scp="$3"
out_dir="$4"

shift 4
while [ $# -gt 0 ]
do
    case $1 in
        --segments_file)
            segments_file="$1 $2"
            shift
            ;;
        --file_type)
            file_type="$1 $2"
            shift
            ;;
        --mlf)
            mlf="$2"
            shift
            ;;
        --sil_labels)
            sil_labels="$1 $2"
            shift
            ;;
        --remove_stress)
            remove_stress="$1"
            ;;
    esac
    shift
done

source "$setup" || exit 1

mkdir -p $out_dir

# convert ctm label file to clusters file
clusters_file=${out_dir}/clusters.txt
if [ ! -e $clusters_file ]; then
    convert_ctm_to_eval2_cluster --write_sequence $segments_file $ctm_file $clusters_file
fi

# extract cluster segments into directores
if [ ! -e $out_dir/.done_extract ]; then
    extract_cluster_segments $file_type $clusters_file $wav_scp $out_dir
    date > $out_dir/.done_extract
    echo "done"
else
    echo "Cluster segments already extracted into directories, skipping...."
fi

# label multiple clusters
if [ ! -z $mlf -a ! -e $out_dir/.done_label ]; then
    label_clusters $remove_stress $sil_labels $mlf $clusters_file $out_dir
    date > $out_dir/.done_label
    echo "done"
else
    echo "Clusters with multiple segments already labeled (Step 1), skipping...."
fi

# label multiple clusters
if [ ! -z $mlf -a ! -e $out_dir/Single/.done_label ]; then
    label_clusters $remove_stress $sil_labels $mlf $clusters_file $out_dir/Single
    date > $out_dir/Single/.done_label
    echo "done"
else
    echo "Clusters with single segment already labeled (Step 2), skipping...."
fi
