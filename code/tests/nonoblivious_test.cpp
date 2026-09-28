#include "../include/NonObliviousJFYan.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

namespace
{
using Table = NonObliviousTable;

std::vector<std::vector<int>> bruteForce(
    const std::vector<Table> &tables,
    const std::vector<int> &parent,
    int root,
    const std::vector<int> &joinColInParent,
    const std::vector<int> &joinColInChild,
    const std::vector<int> &tableKeys)
{
    (void)root;
    std::vector<std::vector<int>> output;
    std::vector<int> selected(tables.size(), 0);

    const auto enumerate = [&](const auto &self, int tableId) -> void
    {
        if (tableId == static_cast<int>(tables.size()))
        {
            for (int child = 0; child < static_cast<int>(tables.size()); ++child)
            {
                if (parent[child] == -1)
                    continue;
                const auto &p = tables[parent[child]][selected[parent[child]]];
                const auto &c = tables[child][selected[child]];
                if (p[joinColInParent[child]] != c[joinColInChild[child]])
                    return;
            }

            std::vector<int> row;
            for (int node = 0; node < static_cast<int>(tables.size()); ++node)
            {
                const auto &source = tables[node][selected[node]];
                row.insert(row.end(), source.begin(), source.begin() + tableKeys[node]);
            }
            output.push_back(std::move(row));
            return;
        }

        for (int rowId = 0; rowId < static_cast<int>(tables[tableId].size()); ++rowId)
        {
            selected[tableId] = rowId;
            self(self, tableId + 1);
        }
    };

    enumerate(enumerate, 0);
    std::sort(output.begin(), output.end());
    return output;
}

void check(const std::vector<Table> &tables,
           const std::vector<int> &parent,
           int root,
           const std::vector<int> &joinColInParent,
           const std::vector<int> &joinColInChild,
           const std::vector<int> &tableKeys)
{
    auto expected = bruteForce(tables, parent, root, joinColInParent, joinColInChild, tableKeys);
    auto actual = NonObliviousJFYan(tables, parent, root, joinColInParent,
                                    joinColInChild, tableKeys);
    assert(actual.status == NonObliviousJFYanOk);
    assert(actual.exactRows == static_cast<long long>(expected.size()));
    std::sort(actual.output.begin(), actual.output.end());
    assert(actual.output == expected);
}
} // namespace

int main()
{
    {
        std::vector<Table> tables = {
            {{15, 10, 3}, {8, 7, 5}, {9, 10, 1}, {8, 9, 3}},
            {{10, 8}, {11, 9}, {12, 8}, {13, 15}},
            {{7, 15}, {7, 16}, {10, 17}, {9, 18}},
            {{1, 1}, {3, 4}, {3, 2}, {5, 1}},
            {{1, 1}, {3, 2}, {3, 3}, {1, 4}},
            {{1, 5}, {2, 6}, {4, 7}, {4, 8}}};
        check(tables,
              {-1, 0, 0, 0, 3, 3},
              0,
              {-1, 0, 1, 2, 0, 1},
              {-1, 1, 0, 0, 0, 0},
              {3, 2, 2, 2, 2, 2});
    }

    std::mt19937 generator(7);
    std::uniform_int_distribution<int> value(0, 3);
    for (int trial = 0; trial < 200; ++trial)
    {
        std::vector<Table> tables(4);
        for (Table &table : tables)
        {
            for (int row = 0; row < 3; ++row)
                table.push_back({value(generator), value(generator), trial * 10 + row});
        }

        // The root is node 2: 2 -> 0 -> 1 and 2 -> 3.
        check(tables,
              {2, 0, -1, 2},
              2,
              {1, 0, -1, 0},
              {0, 1, -1, 1},
              {3, 3, 3, 3});
    }

    {
        std::vector<Table> tables = {{{1, 2}}, {}};
        check(tables, {-1, 0}, 0, {-1, 0}, {-1, 0}, {2, 0});
    }

    {
        std::vector<Table> tables = {{{1}}, {{1}, {1}}};
        auto limited = NonObliviousJFYan(tables, {-1, 0}, 0,
                                         {-1, 0}, {-1, 0}, {1, 1}, 1);
        assert(limited.status == NonObliviousJFYanOutputTooLarge);
        assert(limited.exactRows == 2);
    }

    {
        std::vector<Table> tables = {{{1}}, {{1}}};
        auto invalid = NonObliviousJFYan(tables, {1, 0}, 0,
                                         {-1, 0}, {-1, 0}, {1, 1});
        assert(invalid.status == NonObliviousJFYanInvalidInput);
    }

    // Packed balanced-tree workload layout used by App.cpp: a complete binary
    // tree of depth two, with one-to-one keys and two final output tuples.
    {
        std::vector<Table> tables = {
            {{10, 20, 100}, {11, 21, 101}},
            {{10, 30, 40, 110}, {11, 31, 41, 111}},
            {{20, 50, 60, 120}, {21, 51, 61, 121}},
            {{30, 130}, {31, 131}},
            {{40, 140}, {41, 141}},
            {{50, 150}, {51, 151}},
            {{60, 160}, {61, 161}},
        };
        check(tables,
              {-1, 0, 0, 1, 1, 2, 2},
              0,
              {-1, 0, 1, 1, 2, 1, 2},
              {-1, 0, 0, 0, 0, 0, 0},
              {3, 4, 4, 2, 2, 2, 2});
    }

    std::cout << "NonObliviousJFYan tests passed\n";
    return 0;
}
