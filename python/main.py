from nltk.lm.models import Lidstone
from nltk.lm.models import MLE as nltk_MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

nltk_lm = nltk_MLE(2)

ld = Lidstone(1, 2)

text = [["a", "b", "d"], ["a", "b", "c", "d", "c", "e", "f"]]

train, vocab = padded_everygram_pipeline(2, text)
nltk_lm.fit(train, vocab)
train, vocab = padded_everygram_pipeline(2, text)

ld.fit(train, vocab)

print(ld.perplexity([("a", "b"), ("b", "c")]))
print(nltk_lm.perplexity([("a", "b"), ("b", "c")]))
