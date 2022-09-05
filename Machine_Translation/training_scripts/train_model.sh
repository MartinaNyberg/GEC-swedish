#!/bin/bash -l

#SBATCH -A uppmax2020-2-2 # project no.
#SBATCH -M snowy # cluster name
#SBATCH -t 3-00:00:00 # time
#SBATCH -J train # job name
#SBATCH -p node
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1


# load modules and set the environment
module load python/3.8.7
module load gcc
source /home/martiny/GEC387/bin/activate 

#cat /proj/uppmax2021-2-31/martina/data/train.clean.bpe.err /proj/uppmax2021-2-31/martina/data/train.clean.bpe.cor | ./marian/build/marian-vocab > /proj/uppmax2021-2-31/martina/data/vocab.yml

./marian/build/marian \
    --devices 0 --sync-sgd \
    --model /proj/uppmax2021-2-31/martina/model/model.npz --type transformer \
    --train-sets /proj/uppmax2021-2-31/martina/data/train.clean.bpe.err /proj/uppmax2021-2-31/martina/data/train.clean.bpe.cor \
    --vocabs /proj/uppmax2021-2-31/martina/data/vocab.yml /proj/uppmax2021-2-31/martina/data/vocab.yml \
    --guided-alignment /proj/uppmax2021-2-31/martina/data/corpus.align \
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
    --valid-sets /proj/uppmax2021-2-31/martina/data/dev.bpe.err /proj/uppmax2021-2-31/martina/data/dev.bpe.cor \
    --valid-script-path 'bash /proj/uppmax2021-2-31/martina/scripts/validate.sh' \
    --valid-translation-output /proj/uppmax2021-2-31/martina/data/validation-output-after-{U}-updates-{T}-tokens.txt --quiet-translation \
    --valid-mini-batch 64 \
    --log /proj/uppmax2021-2-31/martina/model/train.log --valid-log /proj/uppmax2021-2-31/martina/model/valid.log \
    --overwrite --keep-best \

deactivate
