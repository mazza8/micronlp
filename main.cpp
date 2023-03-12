#include "micronlp/metrics.h"
#include "micronlp/lm.h"
#include <vector>

using namespace std;

int main() {
    vector < vector < string >> text{{"a", "b", "d"},
                                     {"a", "b", "c", "d", "c", "e", "f"}};


    MLE lm(3);
    lm.fit(text);
    cout << lm.perplexity({{"a", "b"},
                           {"b", "c"}}) << endl;
}