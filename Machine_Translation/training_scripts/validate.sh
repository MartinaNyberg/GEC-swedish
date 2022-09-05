cat $1 | sed 's/\@\@ //g' > PATH_TO_TRANS_OUTPUT/trans.out
python3 PATH_TO_GLEU_SCRIPTS/compute_gleu.py -r PATH_TO_DEV_DATA/dev.bpe.cor -s PATH_TO_DEV_DATA/dev.bpe.err \
    -o OUTPUT_PATH/trans.out

