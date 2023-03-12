#include <vector>
#include <set>
#include <map>

using namespace std;

class MLE {
public:
    int order;
    float gamma;
    string unk_label;
    set<string> vocabulary;

    map<string, map<string, int>> ngrams_counts;

    MLE(int order, float gamma = 0, string unk_label = "<UNK>") {
        this->order = order;
        this->gamma = gamma;
        this->unk_label = unk_label;
    }

    void fit(vector<vector<string>> corpus) {
        for (vector<string> &sentence: corpus) {
            vector<string> padded_sentence = pad_sentence(sentence);
            vocabulary.insert(padded_sentence.begin(), padded_sentence.end());
            for (long unsigned int o = 1; o <= order; o++) {
                for (long unsigned int i = 0; i <= padded_sentence.size() - o; i++) {
                    vector<string> current_ngram = {padded_sentence.begin() + i, padded_sentence.begin() + i + o};
                    ngrams_counts[join({current_ngram.begin(), current_ngram.end() - 1})][current_ngram[(
                            current_ngram.size() -
                            1)]] += 1;
                }
            }
        }
    }

    float perplexity(vector<vector<string>> ngrams) {
        float score = 0;
        for (auto ngram: ngrams) {
            string context = join({ngram.begin(), ngram.end() - 1});
            float norm_count = 0;
            for (auto word: ngrams_counts[context]) {
                norm_count += word.second;
            }
            float word_count = ngrams_counts[context][ngram[(ngram.size() - 1)]];
            float ngram_score = (word_count + gamma) / (norm_count + (vocabulary.size() + 1) * gamma);
            score += (ngram_score) ? log2(ngram_score) : -INFINITY;
        }
        return pow(2, -score / ngrams.size());
    }

private:
    string join(vector<string> strings) {
        string out;
        for (auto string: strings) {
            out += string;
        }
        return out;
    }

    vector<string> pad_sentence(vector<string> sentence) {
        sentence.insert(sentence.begin(), order - 1, "<s>");
        sentence.insert(sentence.end(), order - 1, "</s>");
        return sentence;
    }
};