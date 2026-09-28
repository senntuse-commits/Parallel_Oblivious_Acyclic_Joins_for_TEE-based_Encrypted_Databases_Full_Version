#pragma once

#include <vector>

using NonObliviousTable = std::vector<std::vector<int>>;

enum NonObliviousJFYanStatus
{
    NonObliviousJFYanOk = 0,
    NonObliviousJFYanInvalidInput,
    NonObliviousJFYanCountOverflow,
    NonObliviousJFYanOutputTooLarge
};

struct NonObliviousJFYanResult
{
    NonObliviousTable output;
    long long exactRows = 0;
    double setupMs = 0.0;
    double bottomUpCriticalMs = 0.0;
    double materializeCriticalMs = 0.0;
    double parallelEstimateMs = 0.0;
    NonObliviousJFYanStatus status = NonObliviousJFYanOk;
};

// Bottom-up contributions and top-down positions, with one output allocation.
// Uses hash indexes, data-dependent branches and direct output[pos] writes;
// access patterns are not hidden.
NonObliviousJFYanResult NonObliviousJFYan(
    const std::vector<NonObliviousTable> &tables,
    const std::vector<int> &parent,
    int root,
    const std::vector<int> &joinColInParent,
    const std::vector<int> &joinColInChild,
    const std::vector<int> &tableKeys,
    int maxOutputRows = 2147483647);
