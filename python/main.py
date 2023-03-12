from nltk.lm.models import StupidBackoff
from nltk.lm.preprocessing import padded_everygram_pipeline

ld = StupidBackoff(order=2)

text = [["a", "b", "d"], ["a", "b", "c", "d", "c", "e", "f"]]

train, vocab = padded_everygram_pipeline(2, text)

ld.fit(train, vocab)

print(ld.perplexity([("d", "b")]))
