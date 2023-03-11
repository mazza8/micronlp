import timeit

from nltk.lm.models import MLE as nltk_MLE

from micronlp import MLE as micronlp_MLE
from micronlp import edit_distance

print(edit_distance("rain", "shine"))

lm = micronlp_MLE(2)

text = [["a", "b", "d"], ["a", "b", "c", "d", "c", "e", "f"]]

print(timeit.timeit(lambda: lm.fit(text), number=1000))

print(timeit.timeit(lambda: lm.perplexity([["a", "b"],
                                           ["b", "c"]]), number=10000))

lm = nltk_MLE(2)

from nltk.lm.preprocessing import padded_everygram_pipeline


def temp():
    train, vocab = padded_everygram_pipeline(2, text)
    lm.fit(train, vocab)


print(timeit.timeit(lambda: temp()
                    , number=1000))
print(timeit.timeit(lambda: lm.perplexity([("a", "b"), ("b", "c")]), number=10000))
