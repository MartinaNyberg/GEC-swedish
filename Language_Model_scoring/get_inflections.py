import xml.etree.ElementTree as ET
import spacy_udpipe
import argparse

spacy_udpipe.download("sv") 
nlp = spacy_udpipe.load("sv")

def read_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf8") as vocab_io:
        for i, line in enumerate(vocab_io):
            try:
                word = line.strip().split()[0]
                vocab.append(word)
            except:
                pass
    vocab = set(vocab)
    return vocab

def get_forms(token, lemma, tree):
    """Get inflections for a given token."""
    all_forms = []

    for e in tree.iter(tag='LexicalEntry'):
        tree_lemma = e.find('Lemma')
        formrep = tree_lemma.find('FormRepresentation')
        values = formrep.findall('feat')
        if values[0].attrib['val'] == lemma:
            forms = e.findall('WordForm')
            all_forms.append(forms)

    if len(all_forms) == 0:
        return None

    form_set = set()
    for forms in all_forms:
        for a in forms:
            val = a.find('feat')
            form_set.add(val.attrib['val'])
    form_set = {tok for tok in form_set if tok!=token and not tok.endswith("-") and not tok.startswith("-")}

    return form_set

def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', help="Path to vocabulary text file", required=True)
    parser.add_argument('-s', '--saldo', help="Path to saldo morphology lexicon (xml)", required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_user_args()
    vocab = read_vocab(args.vocab)
    tree = ET.parse(args.saldo)

    for word in vocab:
        doc = nlp(word)
        lemma = [word.lemma_ for word in doc][0]
        replacements = get_forms(word, lemma, tree)
        if replacements:
            try:
                print(word, end='\t')
                print("\t".join(replacements))
            except:
                pass

