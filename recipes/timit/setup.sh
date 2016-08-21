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
#db_path=/export/corpora/LDC/LDC93S1/timit/TIMIT
db_path=/mnt/matylda2/data/TIMIT/timit
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
queues="all.q@@stable"
parallel_opts="-q $queues -l scratch1=0.3,matylda5=0.3"

## SGE - CLSP ##
# queues="all.q"
# parallel_opts="-q $queues -l arch=*64"

############################################################################
# Splitting of the data base (train/dev/test set).                         #
############################################################################
train_keys="$root/data/train.keys"
test_keys="$root/data/test.keys"

############################################################################
# Features settings.                                                       #
############################################################################
scp=${root}/data/all.scp
fea_ext='fea'
fea_type=mfcc
fea_dir=$root/$fea_type
fea_conf=$root/conf/$fea_type.cfg

############################################################################
# Model settings.                                                          #
############################################################################
sil_ngauss=10
concentration=1
truncation=100
nstates=3
ncomponents=2
alpha=3
kappa=5
a=3
b=3
model_type=ploop_${fea_type}_c${concentration}_T${truncation}_sil${sil_ngauss}_s${nstates}_g${ncomponents}_a${a}_b${b}

############################################################################
# Language model training.                                                 #
############################################################################
lm_params=".5,1:.5,1"
lm_train_niter=5
lm_weight=1
ac_weight=1

############################################################################
# Lattices and counts generation.                                          #
############################################################################
beam_thresh=0.0
penalty=-1
gscale=1
latt_count_order=3
sfx=bt${beam_thresh}_p${penalty}_gs${gscale}
conf_latt_dir=${root}/${model_type}/conf_lattices

