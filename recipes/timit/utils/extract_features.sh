#!/usr/bin/env bash

#
# Extract the features for a audio file. The kind of features to extract 
# has to be specified in the setup.sh script.
#

if [ $# -ne 3 ] 
    then
    echo usage: $0 "<setup.sh> <audio_file> <key>"
    exit 1
fi

setup=$1
audio=$2
key=$3

# Load the configuration.
source $setup || exit 1

# Output features file
features=$fea_dir/$key.${fea_ext}

# Create a specific temp directory for the script.  
tmp=`mktemp -d`

# Check audio file for requested times /path/to/file/filename.ext[time1[time2
t_start=`echo $audio | cut -d'[' -sf2`
t_end=`echo $audio | cut -d'[' -sf3`
audio=`echo $audio | cut -d'[' -f1`

# If the audio file has NIST SPHERE format convert it to WAV format.
if test -n "`file $audio | grep SPHERE`" 
    then
    if [ ! -z $t_start ] && [ ! -z $t_end ]
        then
        sph2pipe -f wav -t ${t_start}:${t_end} $audio $tmp/audio.wav
    else
        sph2pipe -f wav $audio $tmp/audio.wav 
    fi
elif test -n "`file $audio | grep symbolic`"
    then
    audio=`readlink -f $audio`
    if [ ! -z $t_start ] && [ ! -z $t_end ]
        then
        sph2pipe -f wav -t ${t_start}:${t_end} $audio $tmp/audio.wav
    else
        sph2pipe -f wav $audio $tmp/audio.wav
    fi
else
    cp $audio $tmp/audio.wav
    chmod u+w $tmp/audio.wav
fi


# If the features are BN then we need to process the data in 2 steps:
#   * extract the FBANK features
#   * extract the BN
# otherwise we extract the features with HCopy.
if [ "$fea_type" = "bn" ] 
    then
    $bn_fbank -Nbands 24 $tmp/audio.wav $tmp/$key || exit 1
    echo $tmp/${key} > $tmp/list
    $extract_bn \
        -l $fea_dir \
        -y $fea_ext \
        -C $fea_conf \
        -H $bn_network \
        -S $tmp/list \
        --STARTFRMEXT=15 --ENDFRMEXT=15 || exit 1

else
    HCopy -C $fea_conf $tmp/audio.wav $features || exit 1
fi

# Clean up the tmp directory
rm -fr $tmp

