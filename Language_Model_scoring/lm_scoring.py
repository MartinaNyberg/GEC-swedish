import spacy_udpipe
import argparse

from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx

def main(args):
    res_dict = loadResources(args)
    # Create output file
    out_sents = open(args.out, "w")

    # Process each tokenized sentence
    with open(args.input_sents) as sents:
      for i, sent in enumerate(sents):
          upper = False
          if sent.isupper(): 
              sent = sent.lower()
              upper = True
          sent = sent.strip()
          tok_sent = sent.split()
          if len(tok_sent) == 1:
              tok_sent.append(".")
              sent = sent + " ."
          # If the line is empty, preserve the newline in output and continue
          if not tok_sent: 
              out_sents.write("\n")
              continue

          sent = processSent(sent, res_dict, args)
          sent = processSent(sent, res_dict, args)
          sent = " ".join(sent)
          if upper: sent = sent.upper()

          # Write corrected sentence to file.
          out_sents.write(sent+"\n")

def loadWordFormDict(file_name):
  form_dict = {}
  with open(file_name, "r", encoding="utf8") as f:
    for line in f:
      words = line.split("\t")
      form_dict[words[0]] = words[1:]
    return form_dict

def loadResources(args):
    lm = scorer
    nlp = spacy_udpipe.load("sv")
    sv = loadWordFormDict(args.spelling)
    sv_inf = loadWordFormDict(args.lexicon)
	# List of common determiners
    det = {"en", "ett", "den", "det", "de"}
	# List of common prepositions
    prep = {"i", "på", "för", "till", "med", "om", "av", "inom", "mellan", "med", "ur", "från", "under", "vid", "mot", "åt", "före", "efter", "utan", "bakom"}
	# Save the above in a dictionary
    res_dict = {"lm": lm,
				"nlp": nlp,
				"sv": sv,
				"sv_inf": sv_inf,
				"det": det,
				"prep": prep}
    return res_dict

def processSent(sent, res_dict, args):
  proc_sent = processWithSpacy(sent, res_dict["nlp"])

  if isinstance(sent, str):
    sent = sent.split()
  sent = proc_sent.text.split()

  # Probability of sentence before correction
  orig_prob = res_dict["lm"].score_sentences([proc_sent.text])[0]/len(proc_sent)
# Store sentences corrected with each candidate token
  cand_dict = {}
  # Process each token.
  for tok in proc_sent:
      # Spelling
      if tok.text.isalpha() and tok.text in res_dict["sv"]:
          cands = res_dict["sv"][tok.text]
          cands = [cand for cand in cands if "-" not in cand]
          # Only save 10 first cands
          cands = cands[:10]
          if cands: 
              cand_dict.update(generateCands(tok.i, cands, sent, args.threshold))

      # Morphology
      if tok.text in res_dict["sv_inf"]:
        cands = res_dict["sv_inf"][tok.text]
        cands = [cand for cand in cands if "-" not in cand]
        if cands is not None:
          cand_dict.update(generateCands(tok.i, cands, sent, args.threshold))

      # Determiners
      if tok.text in res_dict["det"]:
        cand_dict.update(generateCands(tok.i, res_dict["det"], sent, args.threshold))

      # Prepositions
      if tok.text in res_dict["prep"]:
        cand_dict.update(generateCands(tok.i, res_dict["prep"], sent, args.threshold))

  if len(cand_dict.keys()) > 0:
    cand_sents = list(cand_dict.values())
    joined_cand_sents = [" ".join(sent_list) for sent_list in cand_sents]
    try:
        cand_probs = res_dict["lm"].score_sentences(joined_cand_sents) # Score all new sentences with BERT
    except IndexError:
        print(f"IndexError for sentence: {sent}")
        return sent

    normalized_probs = []
    for i, prob in enumerate(cand_probs):
      normalized_prob = prob/len(cand_sents[i])
      normalized_probs.append(normalized_prob)

    # New version of candidate dictionary containing normalized probabilities
    new_cand_dict = {}
    for i, cand in enumerate(cand_dict.keys()):
      new_cand = cand + (normalized_probs[i], )
      new_cand_dict[new_cand] = cand_dict[cand]

    for tok in proc_sent:
      best_prob = float("-inf")
      best_sent = [] # Keep track of the best sentence
      tok_cands = {}
      for cand_key, cand_sent in new_cand_dict.items():
        if cand_key[0] == tok.i:
          tok_cands[cand_key] = cand_sent
      for cand, snt in tok_cands.items():
        # Check if sentence has higher prob than original
        if cand[-1] > orig_prob*args.threshold and cand[-1] > best_prob:
          best_prob = cand[-1]
          best_cand = cand
          best_sent = snt

      if best_sent:
        sent[tok.i] = best_cand[1]
  return sent
        
def processWithSpacy(sent, nlp):
    if isinstance(sent, list):
        sent = " ".join(sent)
    proc_sent = nlp(sent)
    return proc_sent
	
# Input 1: A token index indicating the target of the correction.
# Input 2: A list of candidate corrections for that token.
# Input 3: The current sentence as a list of token strings.
# Input 4: An error type weight
# Output: A dictionary. Key is a tuple: (tok_id, cand, weight),
# value is a list of strings forming a candidate corrected sentence.
def generateCands(tok_id, cands, sent, weight):
  edit_dict = {}
  for cand in cands:
    if cand.endswith("\n"):
      cand = cand.split("\n")[0]
		  # Copy the input sentence
    new_sent = sent[:]
    try:
        new_sent[tok_id] = cand.rstrip("\n")
        new_sent = list(filter(None, new_sent))
        edit_id = (tok_id, cand, weight)
    except IndexError:
        pass
    if new_sent != sent and len(new_sent) > 0: 
      edit_dict[edit_id] = new_sent
  return edit_dict


# Define and parse program input
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', required=True) # For GPU use
parser.add_argument('-l', '--lexicon', help="The path to a word form lexicon", required=True)
parser.add_argument('-s', '--spelling', help="The path to a spelling lexicon", required=True)
parser.add_argument("input_sents", help="A text file containing 1 tokenized sentence per line.")
parser.add_argument("-o", "--out", help="The output correct text file, 1 tokenized sentence per line.", required=True)
parser.add_argument("-th", "--threshold", help="LM percent improvement threshold. Default: 0.96 requires scores to be at least 4% higher than the original.", type=float, default=0.96)
args = parser.parse_args()
	# Run the program.
ctxs = [mx.gpu(args.device)]
model, vocab, tokenizer = get_pretrained(ctxs, "KB/bert-base-swedish-cased")
scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
main(args)
