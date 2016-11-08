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
## UPB ##
xitsonga_base=/net/nas/walter/Database/ZeroSpeech2015/xitsonga
xitsonga_wavs=${xitsonga_base}/xitsonga_wavs
xitsonga_eval1=${xitsonga_base}/xitsonga_eval1
xitsonga_eval2=${xitsonga_base}/xitsonga_eval2

root=$(pwd -P) 

############################################################################
# SGE settings.                                                            #
############################################################################

# Set your parallel environment. Supported environment are:
#   * local
#   * sge
#   * openccs.
export AMDTK_PARALLEL_ENV="local"

parallel_n_core=32
parallel_profile="--profile $root/path.sh"

## SGE - BUT ## 
#queues="all.q@@stable"

## SGE - CLSP ## 
#queues="all.q"

############################################################################
# Features settings.                                                       #
############################################################################
scp=${root}/data/va.scp
fea_ext='fea'
fea_type=mfcc
fea_dir=$root/$fea_type
fea_conf=$root/conf/$fea_type.cfg

## SGE - BUT ##
#fea_parallel_opts="-q $queues -l scratch1=0.3"

## SGE - CLSP ##
#fea_parallel_opts="-q $queues -l arch=*64"

## OpenCCS - UPB ##
fea_parallel_opts="-t 1m"

############################################################################
# Model settings.                                                          #
############################################################################
sil_ngauss=0
concentration=1
truncation=100
nstates=3
ncomponents=2
alpha=3
kappa=5
a=3
b=3
model_type="ploop_${fea_type}_c${concentration}_T${truncation}_sil${sil_ngauss}_s${nstates}_g${ncomponents}_a${a}_b${b}"
unigram_ac_weight=1.0

############################################################################
# Training settings.                                                       #
############################################################################
train_keys=$root/data/va.keys

## SGE - BUT ## 
#train_parallel_opts="-q $queues -l scratch1=0.2,matylda5=0.3"

## SGE - CLSP ## 
#train_parallel_opts="-q $queues -l arch=*64"

## OpenCCS - UPB ##
train_parallel_opts="-t 40m"

############################################################################
# Labeling settings.                                                       #
############################################################################
label_keys=$root/data/va.keys

## SGE - BUT ## 
#label_parallel_opts="-q $queues -l matylda5=0.3"

## SGE - CLSP ## 
#label_parallel_opts="-q $queues -l arch=*64"

## OpenCCS - UPB ##
label_parallel_opts="-t 30m"

############################################################################
# Posteriors settings.                                                     #
############################################################################
post_keys=$root/data/va.keys

## SGE - BUT ## 
#post_parallel_opts="-q $queues -l matylda5=0.3"

## SGE - BUT ## 
#post_parallel_opts="-q $queues -l arch=*64"

## OpenCCS - UPB ##
post_parallel_opts="-t 20m"

############################################################################
# Lattices and counts generation.                                          #
############################################################################
beam_thresh=250.0
penalty=0
gscale=1
latt_count_order=3
sfx=bt${beam_thresh}_p${penalty}_gs${gscale}
conf_latt_dir=${root}/${model_type}/conf_lattices
out_latt_dir=${root}/${model_type}/lattices_${sfx}_${niter}

## SGE - BUT ## 
#latt_parallel_opts="-q $queues -l matylda5=0.3"

## SGE - BUT ## 
#latt_parallel_opts="-q $queues -l arch=*64"

## OpenCCS - UPB ##
latt_parallel_opts="-t 15m"

############################################################################
# Word segmentation                                                        #
############################################################################
word_lm_order="1"
char_lm_order="2 4 6"
addchar_lm_order="0 2 4 6"
ws_niter=150
ws_AMScale="0.2 0.5 0.8"
ws_PruneFactor=10
latticewordsegmentation_threads='1'
latticewordsegmentation_bin='/scratch/owb/Downloads/LatticeWordSegmenation/build/LatticeWordSegmentation'
latticewordsegmentation_parallel_opts='-t 24h --res=rset=mem=15G'

############################################################################
# evaluation1 setup                                                        #
############################################################################
eval1_parallel_opts='--res=rset=ncpus=8:mem=60G -t24h'
eval1_tool="${xitsonga_eval1}/eval1 -kl -j 8"

############################################################################
# evaluation2 setup                                                        #
############################################################################
eval2_parallel_opts='--res=rset=ncpus=4 -t 12h'
eval2_tool="${xitsonga_eval2}/xitsonga_eval2 -v -j 4"

############################################################################
# retrainin setup                                                         #
############################################################################
convert_segmentation_parallel_opts='-t 5m'

############################################################################
# Language model training.                                                 #
############################################################################
lm_params=".5,1:.5,1"
lm_train_niter=5
lm_create_niter=5
lm_weight=1
ac_weight=1
