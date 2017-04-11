#!/bin/bash


# Parallel environmnent settings.
# ---------------------------------------------------------------------
# Parallel profile:
#     * default: local machine
#     * jhu_sge: SGE
profile="default"

# Number of jobs.
njobs=4


# Data settings.
# ---------------------------------------------------------------------
# List of features file for the training.
train_fea_list="test.list"


# Model settings.
# ---------------------------------------------------------------------
# Number of units.
n_units=10

# Number of states per unit.
n_states=3

# Training settings.
# ---------------------------------------------------------------------
training_type=vb
training_args="{'epochs': 10}"
#training_type=svb
#training_args="{'epochs': 3, 'batch_size': 2, 'scale':0.1, 'delay':0, 'forgetting_rate':.5}"

# Clean up function. Its main purpose is to kill
# the running jobs if an error occurs.
# ---------------------------------------------------------------------
trap "clean_exit 1" SIGINT
function clean_exit {
    ipcluster stop --profile "$profile"
    exit "$1"
}


# Start the parallel environment.
# ---------------------------------------------------------------------
ipcluster start --profile "$profile" -n "$njobs" --daemonize
sleep 10


# Compute the statistics of the training data.
# ---------------------------------------------------------------------
#python get_data_stats.py $train_fea_list stats.bin || clean_exit 1


# Build the phone loop model.
# ---------------------------------------------------------------------
#python create_phone_loop.py \
#    --n_units "$n_units" \
#    --n_states "$n_states" \
#    stats.bin ploop.bin || clean_exit 1


# Train the phone loop model.
# ---------------------------------------------------------------------
python train_phone_loop.py \
    --training "$training_type" \
    --train_args "$training_args" \
    stats.bin "$train_fea_list" ploop.bin new_ploop.bin || clean_exit 1


# Stop the parallel environment and exit.
# ---------------------------------------------------------------------
clean_exit 0

