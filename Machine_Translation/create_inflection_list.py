import xml.etree.ElementTree as ET
import spacy_udpipe
import argparse

spacy_udpipe.download("sv") 
nlp = spacy_udpipe.load("sv")

def read_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf8") as vocab_input:
        for i, line in enumerate(vocab_input):
            try:
                word = line.strip().split('\t')[0]
                vocab.append(word)
            except:
                pass
    vocab = set(vocab)
    return vocab

def form_replacements(token, lemma):
        """Get inflections for a given token."""
        pos_conversions = {"nn":"NOUN", "av":"ADJ", "vb":"VERB", "pn":"PRON", "al":"DET"}
        pos_tags = []
        all_forms = []

        for e in tree.iter(tag='LexicalEntry'):
            tree_lemma = e.find('Lemma')
            formrep = tree_lemma.find('FormRepresentation')
            values = formrep.findall('feat')
            if values[0].attrib['val'] == lemma:
                pos = values[2].attrib['val']
                if pos in pos_conversions.keys():
                    pos_tags.append(pos_conversions[pos])
                    forms = e.findall('WordForm')
                    all_forms.append(forms)

        if not all_forms:
            return [lemma]

        form_dict = {}
        for i, forms in enumerate(all_forms):
            tok_pos_string = token + "_" + pos_tags[i]
            form_dict[tok_pos_string] = []
            for a in forms[:20]:
                val = a.find('feat')
                form_dict[tok_pos_string].append(val.attrib['val'])
        
            form_dict[tok_pos_string] = [tok for tok in form_dict[tok_pos_string] if tok!=token and not tok.endswith("-") and not tok.startswith("-")]

        return form_dict

def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', help="Path to vocabulary text file", required=True)
    parser.add_argument('-s', '--saldo', help="Path to saldo morphology lexicon (xml)", required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_user_args()
    vocab = read_vocab(args.vocab)

    tree = ET.parse(args.saldo)
    tags = ["NOUN", "ADJ", "VERB", "PRON", "DET"]

    for word in vocab:
        doc = nlp(word)
        if [word.pos_ for word in doc][0] in tags:
            lemma = [word.lemma_ for word in doc][0]
            replacements = form_replacements(word, lemma)

            if isinstance(replacements, dict):
                for variant, reps in replacements.items():   
                    try: 
                        print(variant, end='\t')
                        print("\t".join(reps))
                    except:
                        pass
            else:
                if replacements[0] != word and replacements[0].isalpha():
                    try:
                        word_pos = word + "_" + [word.pos_ for word in doc][0]
                        print(word_pos, end='\t')
                        print(replacements[0])
                    except:
                        pass

