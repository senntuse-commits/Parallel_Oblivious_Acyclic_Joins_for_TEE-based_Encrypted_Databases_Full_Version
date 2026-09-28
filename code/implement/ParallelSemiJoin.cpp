#include "../include/ParallelSemiJoin.h"
#include "../include/ParallelBitonicSort.h"
#include "../include/Aggtree.h"
#include "../include/ORcompact.h"
#include "../include/PrimitiveProfile.h"
#include "../include/sgx_profile.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace
{
struct FlatSemiTable
{
    int rows = 0;
    int cols = 0;
    vector<int> data;

    FlatSemiTable() = default;
    FlatSemiTable(int r, int c) : rows(r), cols(c), data((size_t)r * c, 0) {}

    inline int &at(int r, int c)
    {
        return data[(size_t)r * cols + c];
    }

    inline const int &at(int r, int c) const
    {
        return data[(size_t)r * cols + c];
    }
};

int selectSemi(bool cond, int a, int b)
{
    const unsigned int mask = 0u - static_cast<unsigned int>(cond);
    return static_cast<int>((static_cast<unsigned int>(a) & mask) |
                            (static_cast<unsigned int>(b) & ~mask));
}

FlatSemiTable sortFlatSemiTableIdx(FlatSemiTable table, const vector<int> &keys, bool direction)
{
    PrimitiveProfileScope primitiveScope(PrimitiveSort);
    const int n = table.rows;
    if (n == 0)
        return table;
    int length = 1;
    while (length < n)
        length <<= 1;
    table.data.resize(static_cast<size_t>(length) * table.cols, 0);
    vector<int> padding(length, 1);
    for (int i = 0; i < n; ++i)
        padding[i] = 0;
    for (int block = 2; block <= length; block <<= 1)
    {
        for (int stride = block >> 1; stride > 0; stride >>= 1)
        {
#pragma omp parallel for schedule(static)
            for (int pair = 0; pair < length / 2; ++pair)
            {
                const int left = (pair / stride) * (stride * 2) + pair % stride;
                const int right = left + stride;
                bool less = padding[left] < padding[right];
                bool greater = padding[left] > padding[right];
                bool equal = padding[left] == padding[right];
                for (int col : keys)
                {
                    const int a = table.at(left, col), b = table.at(right, col);
                    less = less | (equal & (direction ? a < b : a > b));
                    greater = greater | (equal & (direction ? a > b : a < b));
                    equal = equal & (a == b);
                }
                const bool exchange = ((left & block) == 0) ? greater : less;
                const unsigned int mask = 0u - static_cast<unsigned int>(exchange);
                for (int col = 0; col < table.cols; ++col)
                {
                    const unsigned int a = static_cast<unsigned int>(table.at(left, col));
                    const unsigned int b = static_cast<unsigned int>(table.at(right, col));
                    const unsigned int diff = (a ^ b) & mask;
                    table.at(left, col) = static_cast<int>(a ^ diff);
                    table.at(right, col) = static_cast<int>(b ^ diff);
                }
                const int diff = (padding[left] ^ padding[right]) & static_cast<int>(mask);
                padding[left] ^= diff;
                padding[right] ^= diff;
            }
        }
        if (block == length)
            break;
    }
    table.data.resize(static_cast<size_t>(n) * table.cols);
    return table;
}
}
// valueCol = -1 -> compute degree, else aggregate value
// R: parent node S: child node
// Semi-join: R ⋉ S = {r ∈ R | ∃s ∈ S, r.key = s.key}
// joinCol_1: join column index in R
// joinCol_2: join column index in S
ParallelSemiJoin::ParallelSemiJoin(const Table &R, int joinCol_1, const Table &S, int joinCol_2, int valueCol)
    : R(R), S(S), joinCol_1(joinCol_1), joinCol_2(joinCol_2), valueCol(valueCol) {}

void ParallelSemiJoin::execute()
{
    double profStart = sgxProfileNowMs();
    // cout << value.empty() << endl;
    int n = (int)R.size() + (int)S.size();
    const int rCols = R.empty() ? 0 : (int)R[0].size();
    const int sCols = S.empty() ? 0 : (int)S[0].size();
    const bool hasValue = (valueCol != -1);

    // T(key, tid, [val], payload..., orig_idx), tid = 1 => R, tid = 0 => S.
    const int T_KEY = 0;
    const int T_TID = 1;
    const int T_VAL = 2;
    const int payloadBase = hasValue ? 3 : 2;
    const int rLogicalCols = payloadBase + (rCols - 1) + 1;
    const int sPayloadCols = sCols - 1 - (hasValue ? 1 : 0);
    const int sLogicalCols = payloadBase + sPayloadCols + 1;
    const int tCols = std::max(rLogicalCols, sLogicalCols);
    const int T_ORIG = tCols - 1;

    FlatSemiTable T(n, tCols);
    vector<int> B(n, 0), Sc(n, 0), value(n, 0);
    // T(key,tid,val,...)
    // ======== process R ========
#pragma omp parallel for
    for (int i = 0; i < (int)R.size(); i++)
    {
        T.at(i, T_KEY) = R[i][joinCol_1];
        T.at(i, T_TID) = 1;
        if (hasValue)
            T.at(i, T_VAL) = 0;

        int out = payloadBase;
        for (int j = 0; j < (int)R[i].size(); j++)
        {
            if (j == joinCol_1)
                continue;
            T.at(i, out++) = R[i][j];
        }
        T.at(i, T_ORIG) = i;
    }

#pragma omp parallel for
    for (int i = 0; i < (int)S.size(); i++)
    {
        int row = (int)R.size() + i;
        T.at(row, T_KEY) = S[i][joinCol_2];
        T.at(row, T_TID) = 0;
        if (hasValue)
            T.at(row, T_VAL) = S[i][valueCol];

        int out = payloadBase;
        for (int j = 0; j < (int)S[i].size(); j++)
        {
            if (j == joinCol_2)
                continue;
            if (hasValue && j == valueCol)
                continue;
            T.at(row, out++) = S[i][j];
        }
        T.at(row, T_ORIG) = INT_MAX;
    }
    double profBuildT = sgxProfileNowMs();

    // T(key,tid,val,...,idx)
    T = sortFlatSemiTableIdx(std::move(T), {T_KEY, T_TID}, true);
    double profSortT = sgxProfileNowMs();

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        // T.tid == S
        // Cs[i] = (T[i][1] == 0);
        value[i] = !hasValue ? (T.at(i, T_TID) == 0) : T.at(i, T_VAL);
        if (i > 0)
        {
            B[i] = (T.at(i, T_KEY) == T.at(i - 1, T_KEY));
        }
    }
    double profMark = sgxProfileNowMs();

    Sc = Aggtree(value, B, 0).PrefixTreeRun();
    double profAgg = sgxProfileNowMs();

    // collect results: all R rows kept, dummy rows for S (idx=INT_MAX)
    const int col_size_r = rCols;
    const int dummy_size = col_size_r + 2;
    FlatSemiTable flatResult(n, dummy_size);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        bool isR = (T.at(i, T_TID) == 1);
        const bool matched = (Sc[i] > 0) & isR;
        int payloadIndex = payloadBase;
        for (int j = 0; j < col_size_r; ++j)
        {
            const int cell = j == joinCol_1 ? T.at(i, T_KEY) : T.at(i, payloadIndex++);
            flatResult.at(i, j) = selectSemi(matched, cell, 0);
        }
        flatResult.at(i, col_size_r) = selectSemi(matched, Sc[i], 0);
        flatResult.at(i, col_size_r + 1) = selectSemi(isR, T.at(i, T_ORIG), INT_MAX);
    }
    double profBuildResult = sgxProfileNowMs();
    // Sort by original R index (last column) ascending to restore R row order
    flatResult = sortFlatSemiTableIdx(std::move(flatResult), {col_size_r + 1}, true);
    double profSortResult = sgxProfileNowMs();

    result.assign(R.size(), vector<int>(col_size_r + 1));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)R.size(); ++i)
    {
        for (int j = 0; j < col_size_r + 1; ++j)
            result[i][j] = flatResult.at(i, j);
    }
    double profConvert = sgxProfileNowMs();

    if (sgxProfileEnabled())
    {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "    [ParallelSemiJoin R=%d S=%d valueCol=%d] buildT=%.2f ms sortT=%.2f ms mark=%.2f ms agg=%.2f ms buildResult=%.2f ms sortResult=%.2f ms convert=%.2f ms total=%.2f ms",
                 (int)R.size(), (int)S.size(), valueCol,
                 profBuildT - profStart,
                 profSortT - profBuildT,
                 profMark - profSortT,
                 profAgg - profMark,
                 profBuildResult - profAgg,
                 profSortResult - profBuildResult,
                 profConvert - profSortResult,
                 profConvert - profStart);
        sgxProfilePrint(buf);
    }
}

void ParallelSemiJoin::executeValueOnly()
{
    double profStart = sgxProfileNowMs();
    int n = (int)R.size() + (int)S.size();
    if (n == 0)
    {
        resultValues.clear();
        return;
    }
    const bool hasValue = (valueCol != -1);

    // T(key, tid, val, orig_idx), tid = 1 => R, tid = 0 => S.
    const int T_KEY = 0;
    const int T_TID = 1;
    const int T_VAL = 2;
    const int T_ORIG = 3;
    FlatSemiTable T(n, 4);
    vector<int> B(n, 0), Sc(n, 0), value(n, 0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)R.size(); ++i)
    {
        T.at(i, T_KEY) = R[i][joinCol_1];
        T.at(i, T_TID) = 1;
        T.at(i, T_VAL) = 0;
        T.at(i, T_ORIG) = i;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)S.size(); ++i)
    {
        int row = (int)R.size() + i;
        T.at(row, T_KEY) = S[i][joinCol_2];
        T.at(row, T_TID) = 0;
        T.at(row, T_VAL) = hasValue ? S[i][valueCol] : 1;
        T.at(row, T_ORIG) = INT_MAX;
    }
    double profBuildT = sgxProfileNowMs();

    T = sortFlatSemiTableIdx(std::move(T), {T_KEY, T_TID}, true);
    double profSortT = sgxProfileNowMs();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        value[i] = selectSemi(T.at(i, T_TID) == 0, T.at(i, T_VAL), 0);
        if (i > 0)
            B[i] = (T.at(i, T_KEY) == T.at(i - 1, T_KEY));
    }
    double profMark = sgxProfileNowMs();

    Sc = Aggtree(value, B, 0).PrefixTreeRun();
    double profAgg = sgxProfileNowMs();

    // Restore original R order by a fixed network, rather than scattering
    // into resultValues at the secret permutation index T_ORIG.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        T.at(i, T_VAL) = selectSemi(Sc[i] > 0, Sc[i], 0);
    T = sortFlatSemiTableIdx(std::move(T), {T_ORIG}, true);
    resultValues.assign(R.size(), 0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(R.size()); ++i)
        resultValues[i] = T.at(i, T_VAL);
    double profBuildResult = sgxProfileNowMs();

    if (sgxProfileEnabled())
    {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "    [ParallelSemiJoinValue R=%d S=%d valueCol=%d] buildT=%.2f ms sortT=%.2f ms mark=%.2f ms agg=%.2f ms buildResult=%.2f ms sortResult=%.2f ms convert=%.2f ms total=%.2f ms",
                 (int)R.size(), (int)S.size(), valueCol,
                 profBuildT - profStart,
                 profSortT - profBuildT,
                 profMark - profSortT,
                 profAgg - profMark,
                 profBuildResult - profAgg,
                 0.0,
                 0.0,
                 profBuildResult - profStart);
        sgxProfilePrint(buf);
    }
}

/* Sort the table */
Table ParallelSemiJoin::sortTable(Table table, vector<int> keyColumns, bool direction)
{
    ParallelBitonicSorter sorter(std::move(table), std::move(keyColumns), direction);
    sorter.Sorter();
    return sorter.getSortedData();
}

vector<vector<int>> ParallelSemiJoin::getResult()
{
    return std::move(result);
}

vector<int> ParallelSemiJoin::getResultValues()
{
    return std::move(resultValues);
}
