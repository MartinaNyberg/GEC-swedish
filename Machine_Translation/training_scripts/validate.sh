cat $1 | sed 's/\@\@ //g' > /proj/uppmax2021-2-31/martina/data/trans.out
python3 /proj/uppmax2021-2-31/martina/scripts/compute_gleu.py -r /proj/uppmax2021-2-31/martina/data/dev.bpe.cor -s /proj/uppmax2021-2-31/martina/data/dev.bpe.err \
    -o /proj/uppmax2021-2-31/martina/data/trans.out

