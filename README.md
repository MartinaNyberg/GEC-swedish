# GEC-swedish
Grammatical Error Correction for Swedish, done as a Master's Thesis project in Language Technology. Written solely in Python, the code constitutes two approaches to Grammatical Error Correction, found in their respective subfolders:
1. Machine Translation
2. Language Model scoring

### Requirements
- The SweLL gold corpus (Request access at: https://spraakbanken.gu.se/en/projects/swell)
- Aspell for Python, available from https://github.com/WojciechMula/aspell-python
- The Swedish dictionary for Aspell (https://ftp.gnu.org/gnu/aspell/dict/sv/)
- Spacy_udpipe (https://github.com/TakeLab/spacy-udpipe)
- GLEU scripts for evaluation from https://github.com/cnap/gec-ranking

Specific for MT approach:
- Numpy
- NLTK
- marian-nmt (https://marian-nmt.github.io/)
- subword-nmt (https://github.com/rsennrich/subword-nmt)
- fast_align (https://github.com/clab/fast_align) or other alignment tool

Specific for LM scoring approach:
- mlm-scoring (https://github.com/awslabs/mlm-scoring)
- 
## Machine Translation Approach
This approach performs GEC by translating erroneous sentences into corrected versions, using a Transformer model trained on a dataset containing artificially generated errors. The dataset is based on the news portion of the Swedish Culturomics Gigaword Corpus corpus between the years 2010-2015. This corpus can be downloaded from here: https://spraakbanken.gu.se/resurser/gigaword.

In creating a parallel dataset of incorrect-correct sentences, errors are induces into the sentences from the Gigaword Corpus, using the file `error_generation.py`. For this you will need:
- Input data (e.g. the Gigaword Corpus), one sentence per line, cleaned from empty lines.
- A vocabulary file built from the input data, one word per line.
- A file containing word inflections created from the vocabulary by using the file `create_inflection_list.py`. See this file in the `Machine_Translation` folder for instructions. In addition to the vocabulary file, the Saldo morphology lexicon is needed, which can be downloaded here: https://spraakbanken.gu.se/resurser/saldom.
- A file containing word replacements, created through the `create_aspell_replacements.py` file. Run by passing the vocabulary file to the script, and redirect the output to a text file. 

Once the erroneous sentences are generated, these are paired with their original, correct versions. 
