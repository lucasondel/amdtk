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
## CLSP ##
#speech corpora files location
turkish_root=/export/MStuDy/Matthew/JHUWorkshop2016/AUD/tools/amdtk/recipes/babel_turkish_clsp/turkish_data
#train_data_dir=/export/babel/data/105-turkish/release-current-b/conversational/training/audio
train_data_dir=/export/babel/data/105-turkish/release-current-b/conversational/training
train_data_list=/export/babel/data/splits/Turkish_Babel105/train.fullLP.list

#Official DEV data files CLSP
#dev10h_data_dir=/export/babel/data/105-turkish/release-current-b/conversational/dev/audio
dev10h_data_dir=/export/babel/data/105-turkish/release-current-b/conversational/dev
dev10h_data_list=/export/babel/data/splits/Turkish_Babel105/dev.list

# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="--oov <unk>"

lexicon_file=/export/babel/data/105-turkish/release-current-b/conversational/reference_materials/lexicon.txt

# Important Paths for Clustering
path_to_train_ref=$turkish_root/train.ref
path_to_train_decode=$turkish_root/train.decoded
path_to_dev_ref=$turkish_root/dev10h.ref
path_to_dev_decode=$turkish_root/dev10h.decoded
path_to_train_ali=$turkish_root/train.ali
path_to_dev_ali=$turkish_root/dev10h.ali
path_to_lexicon=$turkish_root/lexicon.txt
path_to_phones=$turkish_root/phones.txt

clustering=false
# Set it to true if you want to get a morphemic transcription. You need 
# to have "morfessor" installed and some time. 
morfessor=false
##############

root=$(pwd -P) 
eval_path=${root}/../../../../../EVAL/topics/babel_turkish_clsp
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
model_type=ploop_l${fea_type}_c${concentration}_T${truncation}_s${nstates}_g${ncomponents}

############################################################################
# Training settings.                                                       #
############################################################################
train_fea_dir=$fea_dir
train_dir=$root/$model_type/training
train_keys=$root/data/train.keys
niter=1

## SGE - BUT ## 
#train_parallel_opts="-q $queues -l scratch1=0.2,matylda5=0.3"

## SGE - CLSP ## 
train_parallel_opts="-q $queues -l arch=*64"

############################################################################
# Labeling settings.                                                       #
############################################################################
label_fea_dir=$fea_dir
label_dir=$root/$model_type/labels_$niter
label_keys=$root/data/dev.keys

## SGE - BUT ## 
#label_parallel_opts="-q $queues -l matylda5=0.3"

## SGE - CLSP ## 
label_parallel_opts="-q $queues -l arch=*64"

############################################################################
# Posteriors settings.                                                     #
############################################################################
post_fea_dir=$fea_dir
post_dir=$root/$model_type/posteriors_$niter
post_keys=$root/data/dev.keys

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

