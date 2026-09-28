#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <utility>

using namespace std;

class ObliViatorExpand
{
private:
    vector<vector<int>> T;

public:
    // Zero counts are supported without filtering the input. An explicit
    // publicOutputRows must equal the sum; -1 treats the sum as public (FO).
    // This conservative network implementation uses O((l+output) log^2(l+output))
    // work and O(log^2(l+output)) parallel depth for fixed-width tuples.
    ObliViatorExpand(vector<vector<int>> T_input, vector<int> M_input,
                    int publicOutputRows = -1);

    vector<vector<int>> getResult()
    {
        return std::move(T);
    }
};
