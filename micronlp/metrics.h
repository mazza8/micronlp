#include <iostream>
#include <string>
#include <algorithm>

#include <math.h>

using namespace std;

int edit_distance(string source, string target, int substitution_cost = 1) {
    /*
    Calculate the Levenshtein edit-distance between two strings.
    */

    int n = source.length();
    int m = target.length();

    int dist[n + 1][m + 1] = {0};

    for (int i = 1; i <= n; i++) {
        dist[i][0] = i;
    }
    for (int j = 1; j <= m; j++) {
        dist[0][j] = j;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            dist[i][j] = min(
                    {dist[i - 1][j] + 1, dist[i - 1][j - 1] + (source[i - 1] != target[j - 1] ? substitution_cost : 0),
                     dist[i][j - 1] + 1});
        }
    }
    return dist[n][m];
}

