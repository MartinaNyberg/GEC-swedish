import numpy as np
import random
import nltk
import string
import argparse
import spacy_udpipe

nltk.download("punkt")
spacy_udpipe.download("sv") 
nlp = spacy_udpipe.load("sv")

def load_confusions(filepath):
    conf = dict()
    with open(filepath, 'r', encoding="utf8") as cf:
        for line in cf:
            fields = line.rstrip("\n").split("\t")
            if fields[1:] != ['']:
                conf[fields[0]] = fields[1:]
    return conf


def load_inflections(filepath):
    inflections = dict()
    with open(filepath, 'r', encoding="utf8") as inf:
        for line in inf:
            fields = line.rstrip("\n").split("\t")
            inflections[fields[0]] = fields[1:]
    return inflections


def read_vocab(vocab_path):
    vocab = []
    with open(vocab_path, "r", encoding="utf8") as vocab_input:
        for line in vocab_input:
            try:
                word = line.strip().split('\t')[0]
                vocab.append(word)
            except:
                pass
    vocab = set(vocab)
    return vocab


def preprocess_sent(sent):
    sent = sent.strip()
    sentence = nltk.word_tokenize(sent, language="swedish")
    return sentence


class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence 

    def aspell_replacements(self, word):
        """Get a list of similar words from the aspell dictionary."""
        try:
            replacements = confusions[word]
            if isinstance(replacements, str):
                return [replacements]
            else:
                return replacements
        except:
            return [word]
    
    def form_replacements(self, token, pos):
        """Get a list of inflections for a given token."""
        try:
            tok_pos = token + "_" + pos
            reps = inflections[tok_pos]
            return reps
        except:
            return None
       
    def substitute(self, word):
        """Get a word substitution. The POS decides the source of the substitution."""
        if word in ["en", "ett", "det", "den"]:
            return random.choice([w for w in ["en", "ett", "det", "den"] if w!= word])
        
        doc = nlp([w for w in self.sentence if w])

        try:
            pos = [token.pos_ for token in doc if token.text == word][0]
        except:
            pos = "NONE"

        if pos in ["NOUN", "ADJ", "VERB","PRON", "DET"]:
            suggestions = self.form_replacements(word, pos)
            if not suggestions:
                suggestions = self.aspell_replacements(word)        
        elif pos == "ADP": 
            suggestions = ["i", "på", "för", "till", "med", "om", "av", "inom", "mellan", "med", "ur", "från"]
        else:
            suggestions = self.aspell_replacements(word)
        
        suggestions = [w for w in suggestions if w != word and not w.endswith("-") and not w.startswith("-")]
        if not suggestions:
            # If there are no suggestions, try to return the lemma, else a random word.
            lemma = [token.lemma_ for token in doc if token.text == word]
            if lemma:
                return lemma[0]
            else:
                return random.choice(list(vocab))
        else:
            return random.choice(suggestions)
        
    def insertion(self):
        """Get a random word from the vocabulary."""
        random_word = random.choice(list(vocab))
        return random_word
    
    def induce_errors(self):
        """Induce errors into a sentence."""
        #Sample a value from a normal distribution with mean corresponding to WER in dev data.
        sample = np.random.normal(loc=0.2,scale=0.2,size=1)
        sent_length = len([w for w in self.sentence if w.isalpha()])
        n_words_to_change = round(sample[0] * sent_length) # Number of words to induce errors into.
      
        if n_words_to_change >= 1 and n_words_to_change < len(self.sentence):
            selected = set()
            # Randomly select tokens in the sentence
            selected_words = random.sample(list(enumerate(self.sentence)), n_words_to_change)
            # gives [(index1, word1), (index2, word2), (index3, word3)]                           
            selected.update([word for word in selected_words if word[1].isalpha()]) #Filter selections
            
            if len(selected) < n_words_to_change: # Resample if not enough words after filtering   
                selected_words = random.sample(list(enumerate(self.sentence)), n_words_to_change)
                selected.update([word for word in selected_words if word[1].isalpha()])

            selected_words = list(selected)[:n_words_to_change]
                    
            operations = ["a_substitute", "insert", "swap", "delete"]
            prob_dist = [0.69, 0.08, 0.1, 0.13] # Probabilites of operations based on error types in dev data
            
            error_operations = np.random.choice(operations, len(selected_words), p=prob_dist)  
            error_operations.sort() # apply substitutions first to not mess up indices from 
                                    # other operations
            for n, i_word in enumerate(selected_words):
                i = int(i_word[0])
                word = i_word[1]
                error_operation = error_operations[n] 
                
                if error_operation == "a_substitute":
                    new_word = self.substitute(word)
                    if new_word == word:
                        new_word = new_word[0].upper() + new_word[1:] #Induce capitalization error
                    self.sentence[i] = new_word
                    
                elif error_operation == "insert":
                    insert_word = self.insertion()
                    position = self.sentence.index(word) + 1
                    self.sentence.insert(position, insert_word)
                    
                elif error_operation == "swap":
                    if len(self.sentence) > 1:
                        if self.sentence.index(word) < len(self.sentence)-2:
                            a, b = self.sentence.index(word), self.sentence.index(word)+1
                            self.sentence[a], self.sentence[b] = self.sentence[b], self.sentence[a]
                        
                elif error_operation == "delete":
                    del self.sentence[self.sentence.index(word)]

    def edit_special_letters(self, letters):
        if "ä" in letters:
            index_to_replace = letters.index("ä")
            replacement = random.choice(["a", "e"])
        elif "ö" in letters:
            index_to_replace = letters.index("ö")
            replacement = "o"
        elif "å" in letters:
            index_to_replace = letters.index("å")
            replacement = random.choice(["a", "o"])

        letters[index_to_replace] = replacement
        new_word = "".join(letters)
        return new_word

    def induce_spelling_error(self, n_errors):
        """Induce spelling errors in a sentence."""
        words = self.sentence
        selected_words = set()
        sampled_words = random.sample(words, n_errors)
        selected_words.update([word for word in sampled_words if word.isalpha()]) 
    
        if len(selected_words) < n_errors:
            sampled_words = random.sample(words, n_errors)
            selected_words.update([word for word in sampled_words if word.isalpha()])

        selected_words = list(selected_words)[:n_errors]
    
        operations = ["substitute", "insert", "swap", "delete"]
        prob_dist = [0.7, 0.1, 0.1, 0.1]

        if selected_words:
            for word in selected_words:
                word_index = self.sentence.index(word)
                letters = list(word)
                updated_sentence = False
                modified_word = word
                special = ["å", "ä", "ö"]
                repeated_consonants = False

                # Edit åäö, if present
                if any(x in special for x in letters):
                    modified_word = self.edit_special_letters(letters)
                    self.sentence[word_index] = modified_word
                    updated_sentence = True
                    letters = list(modified_word)
            
                # Check for repeated consonants
                if len(modified_word) > 2:
                    for i in range(len(letters)-1):
                        if letters[i] == letters[i+1] or letters[i] == "c" and letters[i+1] == "k": 
                            del letters[i]
                            repeated_consonants = True
                            break
                    if repeated_consonants == True:
                        modified_word = "".join(letters)
                        self.sentence[word_index] = modified_word
                        updated_sentence = True

                if updated_sentence == False:
                    # If the above operations could not be performed, proceed with operations below.
                    error_operation = np.random.choice(operations, 1, p=prob_dist)
            
                    if error_operation == "substitute":
                        index_to_replace = random.choice(range(len(letters)))
                        replacement = random.choice(string.ascii_lowercase)

                        letters[index_to_replace] = replacement

                    elif error_operation == "insert":
                        if len(letters) > 1:
                            insertion_index = random.choice(range(len(letters)-1))+1
                        else:
                            insertion_index = 1
                        insertion = random.choice(string.ascii_lowercase)
                        letters.insert(int(insertion_index), insertion)

                    elif error_operation == "swap":
                        if len(letters) > 1:
                            swap_index = random.choice(range(len(letters)-1))
                            a, b = swap_index, swap_index+1
                            letters[a], letters[b] = letters[b], letters[a]

                    elif error_operation == "delete":
                        if len(letters) > 2:
                            del letters[-1] # Delete last letter (common in dev data)

                    modified_word = "".join(letters)
                    self.sentence[word_index] = modified_word
        
    def remove_punctuation(self):
        if "," in self.sentence:
            self.sentence.remove(",")
        elif "." in self.sentence:
            self.sentence.remove(".")

            
def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help="Input file containing sentences, one per line.", required=True)
    parser.add_argument('-i', '--infs', help="File with list of inflections for words in input data.", required=True)
    parser.add_argument('-c', '--confs', help="File with list of word replacements from Aspell.", required=True)
    parser.add_argument('-v', '--vocab', help="Vocabulary file from input data.", required=True)
    parser.add_argument('-o', '--outfile', help="Output file for generated erroneous sentences.", required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_user_args()

    inflections = load_inflections(args.infs)
    confusions = load_confusions(args.confs)
    vocab = read_vocab(args.vocab)

    with open(args.data, "r", encoding="utf8") as infile:
        with open(args.outfile, "w", encoding="utf8") as outfile:
            for line in infile:
                tokenized = preprocess_sent(line)
                sent = Sentence(tokenized)
                sent.induce_errors()

                if random.random() < 0.4: # Probability of spelling error in dev data. Can be improved.
                    sent_length = len([w for w in sent.sentence if w.isalpha()])
                    if sent_length > 1:
                        num_errors = [i for i in range(1, sent_length+1)] #number of possible errors.
                        prob_dist = [0.4**i for i in num_errors] #probability of each number of errors
                        apply_n_errors = random.choices(num_errors, weights=tuple(prob_dist), k=1)[0]
                        if apply_n_errors <= len(sent.sentence): 
                            sent.induce_spelling_error(apply_n_errors)

                if "," in sent.sentence or "." in sent.sentence:
                    if random.random() < 0.23: #Probability of punctuation error in dev data.
                        sent.remove_punctuation()

                #Write erroneous sentence to file
                outfile.write(' '.join(sent.sentence))
                outfile.write("\n")
            
