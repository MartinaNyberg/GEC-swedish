#!/bin/bash -l

#SBATCH -A uppmax2020-2-2 # project no.
#SBATCH -M snowy # cluster name
#SBATCH -t 10:00 # time
#SBATCH -J eval # job name
#SBATCH -p node
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1


# load modules and set the environment
module load python/3.8.7
module load gcc
source /home/martiny/GEC387/bin/activate 

cat /proj/uppmax2021-2-31/martina/data/test.bpe.err \
    | ./marian/build/marian-decoder -c /proj/uppmax2021-2-31/martina/model/model.npz.decoder.yml -m /proj/uppmax2021-2-31/martina/model/model.npz -d 0 -b 12 -n 1 -w 12000 \
    | sed "s/\@\@ //g" > /proj/uppmax2021-2-31/martina/data/test.output

python3 /proj/uppmax2021-2-31/martina/scripts/compute_gleu.py -r /proj/uppmax2021-2-31/martina/data/test.cor -s /proj/uppmax2021-2-31/martina/data/test.err \
    -o /proj/uppmax2021-2-31/martina/data/test.output

deactivate
