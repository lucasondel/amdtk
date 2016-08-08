#
# This file defines all the configuration variables for a particular 
# experiment. Set the different paths according to your system. Most of the 
# values predefined here are generic and should yield decent results.
# However, they are most likely not optimal and need to be tuned for each 
# particular data set.
#

############################################################################
# Directories.                                                             #
############################################################################
## BUT ##
#wsj0=/mnt/matylda2/data/WSJ0
#wsj1=/mnt/matylda2/data/WSJ1

## CLSP ##
wsj0=/export/corpora/LDC/LDC93S6A
wsj1=/export/corpora/LDC/LDC94S13A

root=$(pwd -P) 

############################################################################
# SGE settings.                                                            #
############################################################################

# Set your parallel environment. Supported environment are:
#   * local
#   * sge
#   * openccs.
export AMDTK_PARALLEL_ENV="sge"

parallel_n_core=100
parallel_profile="--profile $root/path.sh"

## SGE - BUT ## 
#queues="all.q@@stable"

## SGE - CLSP ## 
queues="all.q"

############################################################################
# Features settings.                                                       #
############################################################################
scp=${root}/data/all.scp
fea_ext='fea'
fea_type=mfcc
fea_dir=$root/$fea_type
fea_conf=$root/conf/$fea_type.cfg

## SGE - BUT ##
#fea_parallel_opts="-q $queues -l scratch1=0.3"

## SGE - CLSP ##
fea_parallel_opts="-q $queues -l arch=*64"

############################################################################
# Model settings.                                                          #
############################################################################
concentration=1
truncation=100
eta=3
nstates=3
alpha=3
ncomponents=2
kappa=5
a=3
b=3
model_type=ploop_l${fea_type}_c${concentration}_T${truncation}_s${nstates}_g${ncomponents}_a${a}_b${b}

############################################################################
# Training settings.                                                       #
############################################################################
train_keys=$root/data/training_unique.keys

## SGE - BUT ## 
#train_parallel_opts="-q $queues -l scratch1=0.2,matylda5=0.3"

## SGE - CLSP ## 
train_parallel_opts="-q $queues -l arch=*64"

############################################################################
# Labeling settings.                                                       #
############################################################################
label_keys=$root/data/all_unique.keys

## SGE - BUT ## 
#label_parallel_opts="-q $queues -l matylda5=0.3"

## SGE - CLSP ## 
label_parallel_opts="-q $queues -l arch=*64"

############################################################################
# Language model training.                                                 #
############################################################################
lm_params=".5,1:.5,1"
lm_train_niter=5
lm_weight=1
ac_weight=1

############################################################################
# Posteriors settings.                                                     #
############################################################################
post_keys=$root/data/all_unique.keys

## SGE - BUT ## 
#post_sge_res="-q $queues -l matylda5=0.3"

## SGE - BUT ## 
post_sge_res="-q $queues -l arch=*64"

############################################################################
# Lattices and counts generation.                                          #
############################################################################
beam_thresh=0.0
penalty=-1
gscale=1
latt_count_order=3
sfx=bt${beam_thresh}_p${penalty}_gs${gscale}
conf_latt_dir=${root}/${model_type}/conf_lattices
out_latt_dir=${root}/${model_type}/lattices_${sfx}_${niter}

## SGE - BUT ## 
#latt_parallel_opts="-q $queues -l matylda5=0.3"

## SGE - BUT ## 
latt_parallel_opts="-q $queues -l arch=*64"

############################################################################
# Word segmentation                                                        #
############################################################################
word_lm_order=1
char_lm_order=4
ws_niter=2
ws_AMScale=1
ws_PruneFactor=16
ws_model_type=ws_w${word_lm_order}_c${char_lm_order}_${ws_niter}_AM${ws_AMScale}_PF${ws_PruneFactor}
lattice_base_path="${root}/${model_type}/unigram_lattices"
ws_base_path=${root}/${ws_model_type}
ws_output_path=$ws_base_path
ws_output_dir=$ws_base_path/KnownN_${word_lm_order}_UnkN_${char_lm_order}
ws_output=${ws_output_dir}/Sentences_Iter_${ws_niter}
output_wordsegmentation_labels=true
ws_output_timing=${ws_output_dir}/TimedSentences_Iter_${ws_niter}
ws_label_dir=${ws_output_dir}/Labels

############################################################################
# Phone language model estimation                                          #
############################################################################
params="0.8,2:0.7,3:0.7,3:0.7,3"
let "order=`echo "$params" |grep -o ":"|wc -l`+1"
fixed="--niter 2 --resample --tokenizer phone --export_fst"
lm_model_type=lm_${order}gram_resample
text_input_file=$ws_output
lm_base_path=${root}/$lm_model_type
lm_output_file="${lm_base_path}/out.lm"
symbols_file=${lm_base_path}/lm.syms

############################################################################
# Lattice rescoring                                                        #
############################################################################
lattices_dir=$lattice_base_path
lattice_type="htk"
rescoring_lm_file=$lm_output_file
rescoring_model_type=rescored_${lm_model_type}
rescoring_base_path=${root}/${rescoring_model_type}
rescoring_lattice_path=${rescoring_base_path}/rescored
rescoring_symbols_file=$symbols_file
rescoring_labels_outdir=${rescoring_base_path}/labels

