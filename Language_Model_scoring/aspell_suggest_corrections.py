import aspell
import sys

s = aspell.Speller('lang', 'sv')
for line in sys.stdin.readlines():
    word = line.strip()
    if s.check(word) == False:
        suggestions = s.suggest(word)
        suggestions = [w for w in suggestions if w != word and not w.endswith("-") and not w.startswith("-")][:20]
        if len(suggestions) > 0:
            print(word + '\t' + '\t'.join(suggestions))

