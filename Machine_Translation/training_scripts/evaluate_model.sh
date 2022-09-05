#!/bin/bash -l

#SBATCH -A PROJECT_NAME # project no.
#SBATCH -M CLUSTER # cluster name
#SBATCH -t 10:00 # time
#SBATCH -J eval # job name
#SBATCH -p node
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1


# load modules and set the environment
module load python/3.8.7
module load gcc
source ENV_NAME 

cat PATH_TO_ERRONEOUS_TEST_FILE \
    | ./marian/build/marian-decoder -c MODEL_PATH/model.npz.decoder.yml -m MODEL_PATH/model.npz -d 0 -b 12 -n 1 -w 12000 \
    | sed "s/\@\@ //g" > DATA_PATH/test.output

python3 PATH_TO_GLEU_SCRIPTS/compute_gleu.py -r /DATA_PATH/test.cor -s DATA_PATH/test.err \
    -o DATA_PATH/test.output

deactivate
