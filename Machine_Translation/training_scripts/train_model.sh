#!/bin/bash -l

#SBATCH -A PROJECT_NAME # project no.
#SBATCH -M CLUSER # cluster name
#SBATCH -t 3-00:00:00 # time
#SBATCH -J train # job name
#SBATCH -p node
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1


# load modules and set the environment
module load python/3.8.7
module load gcc
source ENV_NAME # activate environment

#Create vocabulary from train and test data
cat PATH_TO_ERRONEOUS_TRAIN_DATA PATH_TO_CORRECT_TRAIN_DATA | ./marian/build/marian-vocab > PATH_TO_VOCAB_FILE (.yaml format)

./marian/build/marian \
    --devices 0 --sync-sgd \
    --model PATH_TO_MODEL/model.npz --type transformer \
    --train-sets train.clean.bpe.err train.clean.bpe.cor \
    --vocabs PATH_TO_VOCAB/vocab.yml PATH_TO_VOCAB/vocab.yml \
    --guided-alignment /PATH_TO_ALIGMENT_FILE/corpus.align \
    --guided-alignment-cost ce \
    --guided-alignment-weight 1 \
    --task transformer-base \
    --max-length 1000 \
    --mini-batch-fit --workspace 12000 --maxi-batch 1000 \
    --optimizer-delay 3 \
    --tied-embeddings-all \
    --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
    --transformer-dropout 0.3 \
    --transformer-dropout-attention 0.1 \
    --transformer-dropout-ffn 0.1 \
    --dropout-src 0.3 \
    --dropout-trg 0.1 \
    --label-smoothing 0.1 \
    --exponential-smoothing 0.0001 \
    --early-stopping 10 \
    --after-epochs 5 \
    --beam-size 12 --normalize 1 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics translation \
    --valid-sets dev.bpe.err data/dev.bpe.cor \
    --valid-script-path 'validate.sh' \
    --valid-translation-output PATH/validation-output-after-{U}-updates-{T}-tokens.txt --quiet-translation \
    --valid-mini-batch 64 \
    --log PATH/train.log --valid-log PATH/valid.log \
    --overwrite --keep-best \

deactivate
