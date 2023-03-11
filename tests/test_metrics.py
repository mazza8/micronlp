from nltk.lm.models import MLE

text = [["a", "b", "d"], ["a", "b", "c", "d", "c", "e", "f"]]
from nltk.lm.preprocessing import padded_everygram_pipeline

train, vocab = padded_everygram_pipeline(2, text)
lm = MLE(2)

lm.fit(train, vocab)

print(lm.perplexity([("a", "b"), ("b", "c")]))
