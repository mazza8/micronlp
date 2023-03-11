from nltk.metrics.distance import edit_distance as nltk_edit_distance

from micronlp import edit_distance as mp_edit_distance


def test_edit_distance():
    assert mp_edit_distance("rain", "shine") == nltk_edit_distance("rain", "shine")
