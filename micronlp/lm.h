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
            float ngram_score = compute_score(context, ngram[(ngram.size() - 1)]);
            score += (ngram_score) ? log2(ngram_score) : -INFINITY;
        }
        return pow(2, -score / ngrams.size());
    }

protected:
    float compute_score(string context, string word) {
        float norm_count = 0;
        for (auto context_word: ngrams_counts[context]) {
            norm_count += context_word.second;
        }
        float word_count = ngrams_counts[context][word];
        float ngram_score = (word_count + gamma) / (norm_count + (vocabulary.size() + 1) * gamma);
        return ngram_score;
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

class StupidBackoff : public MLE {
public:

    StupidBackoff(int order, float alpha = 0.4, string unk_label = "<UNK>") : MLE(order, 0, unk_label) {
        this->alpha = alpha;
    }

protected:
    float compute_score(string context, string word) {
        float score = MLE::compute_score(context, word);
        return score == 0 ? alpha * compute_score(context.substr(1, context.size() - 1), word) : score;
    }

private:
    float alpha;
};