#include "micronlp/metrics.h"
#include "micronlp/lm.h"
#include <vector>

using namespace std;

int main() {
    vector < vector < string >> text{{"a", "b", "d"},
                                     {"a", "b", "c", "d", "c", "e", "f"}};

    StupidBackoff lm(2);
    //MLE lm(3);
    lm.fit(text);
    cout << lm.perplexity({{"d", "b"}}) << endl;
}