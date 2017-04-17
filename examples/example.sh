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
train_fea_list="test_fbank.list"


# Model settings.
# ---------------------------------------------------------------------
# GMM Model
# ---------
#model_type="gmm"
#n_components=1
#model_args="{'n_components': $n_components}"

# Phone-loop
# ----------
#model_type="hmm"
#n_units=50
#n_states=3
#model_args="{'n_units': $n_units, 'n_states': $n_states}"

# VAE
# ----------
model_type="vae"
model_args="{'dim_latent': 20, 'n_layers': 1, 'n_units': 200}"


# Training settings.
# ---------------------------------------------------------------------
training_args="{'epochs': 5, 'batch_size':10, 'lrate': 0.001}"
#training_args="{'epochs': 5, 'batch_size': 10, 'scale':1, 'delay':0, 'forgetting_rate':.51}"

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
sleep 20


# Compute the statistics of the training data.
# ---------------------------------------------------------------------
python get_data_stats.py --profile "$profile" $train_fea_list stats.bin || clean_exit 1


# Build the model.
# ---------------------------------------------------------------------
python create_model.py \
    --model_type "$model_type" \
    --model_args "$model_args" \
    stats.bin vae_init.bin || clean_exit 1

# Train the phone loop model.
# ---------------------------------------------------------------------
python train_model.py \
    --profile "$profile" \
    --train_args "$training_args" \
    stats.bin "$train_fea_list" vae_init.bin vae.bin || clean_exit 1


# Transform the features.
# ---------------------------------------------------------------------
python vae_transform_features.py \
    test_fbank.scp vae.bin || clean_exit 1


# Stop the parallel environment and exit.
# ---------------------------------------------------------------------
clean_exit 0

