# GEC-swedish
Grammatical Error Correction for Swedish, done as a Master's Thesis project in Language Technology. Written solely in Python, the code constitutes two approaches to Grammatical Error Correction, found in their respective subfolders:
1. Machine Translation
2. Language Model scoring

### Requirements
- The SweLL gold corpus (Request access at: https://spraakbanken.gu.se/en/projects/swell)
- Aspell for Python, available from https://github.com/WojciechMula/aspell-python
- The Swedish dictionary for Aspell (https://ftp.gnu.org/gnu/aspell/dict/sv/)
- Saldo morphology lexicon (https://spraakbanken.gu.se/resurser/saldom)
- Spacy_udpipe (https://github.com/TakeLab/spacy-udpipe)
- GLEU scripts for evaluation from https://github.com/cnap/gec-ranking

Specific for MT approach:
- Numpy (Python)
- NLTK (Python)
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

Preparing the test and dev data from the SweLL Gold Corpus:
- Extract the source and target sentences for all essays.
- Use a tool for sentence alignment to align the source and target sentences. 
- Create a development set from part of the SweLL data.  Imformation on which essays from the data that were used in the dev set is found in the file `dev_info.txt`. 
- Create a test set from the remaining part of the SweLL data that was not used for development. 

Once the erroneous sentences are generated, these are paired with their original, correct versions to form a parallel dataset that can be used for training a transformer model. 
- Install Marian-nmt (https://marian-nmt.github.io/)
- Preprocess the train, test and dev sets by applying byte-pair encoding through subword-nmt (https://github.com/rsennrich/subword-nmt)
- Create an anlignment file to use in training, from fast_align (https://github.com/clab/fast_align) or other alignment tool
- Adapt the training script `train_model.sh` and validation script `validate.sh` to fit your setup, by specifying the correct file paths and other settings. 
- Once the model is trained, it can be run on the test set, using the script `evaluate_model.sh`. The output will be a GLEU score between 0 and 1.

## Language Model scoring approach

