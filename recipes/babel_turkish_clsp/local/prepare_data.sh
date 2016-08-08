#!/bin/bash
if [ $# -ne 1 ] 
    then
    echo usage: $0 "<setup.sh>"
    exit 1
fi

source $1 || exit 1

if [ ! -d data ] ; then
  mkdir data
fi

# Exit if .done file exists
[ -f data/.done ] && echo "Already Prepared Data." && exit 0

# Prepare clustering
if $clustering ; then  
  ./local/prepare_clustering.sh
fi

# ----------------------- Train Acoustic ---------------------------------------

if [ ! -f data/raw_train_data/.done ] ; then
  ./local/make_corpus_subset.sh $train_data_dir $train_data_list ./data/raw_train_data
  train_data_dir=`readlink -f ./data/raw_train_data`
  touch data/raw_train_data/.done
fi

if [ ! -f data/raw_dev10h_data/.done ] ; then
  ./local/make_corpus_subset.sh "$dev10h_data_dir" "$dev10h_data_list" ./data/raw_dev10h_data
  dev10h_data_dir=`readlink -f ./data/raw_dev10h_data`
  touch data/raw_dev10h_data/.done
fi

if [ ! -f data/train/.done ] ; then
  mkdir -p data/train
  ./local/prepare_acoustic_training_data.pl --vocab $path_to_lexicon --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log
    touch data/train/.done
fi

if [ ! -f data/dev/.done ] ; then  
  mkdir -p data/dev
  ./local/prepare_acoustic_training_data.pl --vocab $path_to_lexicon --fragmentMarkers \-\*\~ \
    $dev10h_data_dir data/dev > data/dev/skipped_utts.log
    touch data/dev/.done
fi

echo "----------------------------------------"
echo "Creating .scp and .keys files"
echo "----------------------------------------"

# Create document ID keys for potential later evaluation
cut -d'.' -f1 data/train/segments | cut -d'_' -f1,2,3,4 | sort -u > data/train_docid.keys
cut -d'.' -f1 data/dev/segments | cut -d'_' -f1,2,3,4 | sort -u > data/test_docid.keys

cat data/train_docid.keys data/test_docid.keys | sort -u -f > data/all_docid.keys

# ---------- This section adds timing and breaks files into smaller utterances. ---------
# Get all keys from the segments files
cut -d' ' -f1 data/train/segments > data/train.keys
cut -d' ' -f1 data/dev/segments > data/dev.keys

# Concatenate train and dev keys to form all keys
cat data/train.keys data/dev.keys > data/all.keys

# Create 2 variables, train_files, and dev_files, which are paths to training
# and dev data with the slashes escaped so that they can be read by sed.
train_files="$(echo $train_data_dir | sed "s/\//\\\\\//g")"
dev_files="$(echo $dev10h_data_dir | sed "s/\//\\\\\//g")"

# Add .sph to the end of each filename in segments, and prepend the absolute path as well.
cut -d' ' -f2 data/train/segments | sed 's/$/.sph/g' | sed "s/^/${train_files}\/audio\//g" > data/train_scp.txt
cut -d' ' -f2 data/dev/segments | sed 's/$/.sph/g' | sed "s/^/${dev_files}\/audio\//g" > data/dev_scp.txt

# Get the start times for each filename in segments
cut -d' ' -f3 data/train/segments > data/train_start.txt
cut -d' ' -f3 data/dev/segments > data/dev_start.txt

# Get the end times for each filename in segments
cut -d' ' -f4 data/train/segments > data/train_end.txt
cut -d' ' -f4 data/dev/segments > data/dev_end.txt

# Paste the start and end times to the end of each file in train_scp.txt. 
# Delimit the file and times by [
paste -d'[' data/train_scp.txt data/train_start.txt data/train_end.txt > data/train_scp_times.txt
paste -d'[' data/dev_scp.txt data/dev_start.txt  data/dev_end.txt > data/dev_scp_times.txt

# Paste the keys to the end of each file separated by :
paste -d':' data/train_scp_times.txt data/train.keys > data/train.scp
paste -d':' data/dev_scp_times.txt data/dev.keys > data/dev.scp

# Concatenate train and dev .scp files to make all.scp
cat data/train.scp data/dev.scp > data/all.scp

# Clean up directory a little
rm data/*.txt

echo "-----------------------------------------------------"
echo "Checking that the .scp and .keys files are correct"
echo "-----------------------------------------------------"

num_keys=$(wc -l data/all.keys | cut -d' ' -f1)
num_scp=$(wc -l data/all.scp | cut -d' ' -f1)

num_train=$(wc -l data/train.keys | cut -d' ' -f1)
num_dev=$(wc -l data/dev.keys | cut -d' ' -f1)
num_total=$(($num_train + $num_dev))

[ $num_keys -eq $num_scp ] && [ $num_total -eq $num_scp ] || exit 1 
  
touch data/.done
echo "Data seems to have been successfully prepared"
