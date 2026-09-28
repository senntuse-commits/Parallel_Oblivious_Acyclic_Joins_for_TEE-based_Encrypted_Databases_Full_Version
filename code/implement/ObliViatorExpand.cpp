#include "../include/ObliViatorExpand.h"
#include "../include/PrimitiveProfile.h"
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace
{
int selectInt(bool chooseFirst, int first, int second)
{
    const uint32_t mask = 0u - static_cast<uint32_t>(chooseFirst);
    return static_cast<int>((static_cast<uint32_t>(first) & mask) |
                            (static_cast<uint32_t>(second) & ~mask));
}

// Always touch both fixed row locations. Never exchange vector pointers.
void networkSort(vector<vector<int>> &rows, const vector<int> &keys)
{
    const int count = static_cast<int>(rows.size());
    const int width = static_cast<int>(rows[0].size());
    for (int block = 2; block <= count; block *= 2)
    {
        for (int stride = block / 2; stride > 0; stride /= 2)
        {
#pragma omp parallel for schedule(static)
            for (int pair = 0; pair < count / 2; ++pair)
            {
                const int left = (pair / stride) * (2 * stride) + pair % stride;
                const int right = left + stride;
                bool less = false, greater = false, equal = true;
                for (int key : keys)
                {
                    const int a = rows[left][key], b = rows[right][key];
                    less = less | (equal & (a < b));
                    greater = greater | (equal & (a > b));
                    equal = equal & (a == b);
                }
                const bool exchange = ((left & block) == 0) ? greater : less;
                const uint32_t mask = 0u - static_cast<uint32_t>(exchange);
                for (int col = 0; col < width; ++col)
                {
                    const uint32_t a = static_cast<uint32_t>(rows[left][col]);
                    const uint32_t b = static_cast<uint32_t>(rows[right][col]);
                    const uint32_t diff = (a ^ b) & mask;
                    rows[left][col] = static_cast<int>(a ^ diff);
                    rows[right][col] = static_cast<int>(b ^ diff);
                }
            }
        }
        if (block == count)
            break;
    }
}
}

ObliViatorExpand::ObliViatorExpand(vector<vector<int>> input, vector<int> counts,
                                 int publicOutputRows)
{
    PrimitiveProfileScope primitiveScope(PrimitiveExpand);
    if (input.size() != counts.size())
        throw invalid_argument("Expansion input/count sizes differ");
    const int n = static_cast<int>(input.size());
    const int width = n == 0 ? 0 : static_cast<int>(input[0].size());
    vector<long long> prefix(n, 0), next(n, 0);
    bool invalid = false;
    for (int i = 0; i < n; ++i)
    {
        invalid = invalid | (counts[i] < 0) | (input[i].size() != static_cast<size_t>(width));
        prefix[i] = counts[i];
    }
    for (size_t distance = 1; distance < input.size(); distance *= 2)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            next[i] = prefix[i] + (static_cast<size_t>(i) >= distance ? prefix[i - distance] : 0);
        prefix.swap(next);
    }
    const long long total = n == 0 ? 0 : prefix.back();
    if (invalid || total < 0 || total > INT_MAX)
        throw invalid_argument("Invalid or overflowing expansion counts");
    const int outputRows = publicOutputRows < 0 ? static_cast<int>(total) : publicOutputRows;
    if (total != outputRows)
        throw invalid_argument("Expansion counts must sum to the public target");
    if (outputRows == 0) // the requested output length is public
        return;
    if (n == 0 || static_cast<long long>(n) + outputRows > (1LL << 29))
        throw length_error("Expansion workspace is too large");

    // Fixed input headers plus public output placeholders. Disabled headers
    // sort after the output; no secret-sized survivor array is constructed.
    const int position = width, kind = width + 1, seen = width + 2;
    int paddedRows = 1;
    while (paddedRows < n + outputRows)
        paddedRows *= 2;
    vector<vector<int>> records(paddedRows, vector<int>(width + 3, 0));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < paddedRows; ++i)
    {
        records[i][position] = INT_MAX;
        records[i][kind] = 2;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        for (int col = 0; col < width; ++col)
            records[i][col] = input[i][col];
        records[i][position] = selectInt(counts[i] > 0,
            static_cast<int>(prefix[i] - counts[i]), outputRows);
        records[i][kind] = 0;
        records[i][seen] = counts[i] > 0;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < outputRows; ++i)
    {
        records[n + i][position] = i;
        records[n + i][kind] = 1;
    }
    networkSort(records, {position, kind});

    // Parallel segmented prefix propagation of the nearest enabled header.
    // All rounds visit the same public workspace locations.
    vector<vector<int>> scan = records, scratch = records;
    for (int distance = 1; distance < paddedRows; distance *= 2)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < paddedRows; ++i)
        {
            const int previous = i >= distance ? i - distance : i;
            const bool inherit = (i >= distance) & (scan[i][seen] == 0);
            for (int col = 0; col < width; ++col)
                scratch[i][col] = selectInt(inherit, scan[previous][col], scan[i][col]);
            scratch[i][seen] = scan[i][seen] | (static_cast<int>(i >= distance) & scan[previous][seen]);
        }
        scan.swap(scratch);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < paddedRows; ++i)
    {
        for (int col = 0; col < width; ++col)
            records[i][col] = scan[i][col];
        records[i][kind] = records[i][kind] != 1;
    }
    networkSort(records, {kind, position});
    T.assign(outputRows, vector<int>(width));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < outputRows; ++i)
        for (int col = 0; col < width; ++col)
            T[i][col] = records[i][col];
}
