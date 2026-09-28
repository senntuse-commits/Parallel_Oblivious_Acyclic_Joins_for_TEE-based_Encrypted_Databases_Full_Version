#include "../include/ParYan.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

namespace
{
using Row = std::vector<int>;

Table bruteForce(const std::vector<Table> &tables,
                 const std::vector<int> &parent,
                 const std::vector<int> &joinParent,
                 const std::vector<int> &joinChild)
{
    Table expected;
    std::vector<int> choice(tables.size(), 0);
    std::function<void(int)> visit = [&](int relation)
    {
        if (relation == static_cast<int>(tables.size()))
        {
            for (int child = 0; child < static_cast<int>(tables.size()); ++child)
            {
                if (parent[child] == -1)
                    continue;
                const Row &p = tables[parent[child]][choice[parent[child]]];
                const Row &c = tables[child][choice[child]];
                if (p[joinParent[child]] != c[joinChild[child]])
                    return;
            }
            Row row;
            for (int r = 0; r < static_cast<int>(tables.size()); ++r)
                row.insert(row.end(), tables[r][choice[r]].begin(), tables[r][choice[r]].end());
            expected.push_back(std::move(row));
            return;
        }
        for (int i = 0; i < static_cast<int>(tables[relation].size()); ++i)
        {
            choice[relation] = i;
            visit(relation + 1);
        }
    };
    visit(0);
    return expected;
}
} // namespace

int main()
{
    std::vector<Table> tables = {
        {{1, 10, 100}, {2, 20, 200}},
        {{7, 1}, {8, 1}, {9, 2}},
        {{10, 5}, {20, 6}},
        {{100, 3}, {200, 4}},
        {{100, 11}, {200, 12}},
        {{3, 21}, {4, 22}},
    };
    const std::vector<int> parent = {-1, 0, 0, 0, 3, 3};
    const std::vector<int> tableKeys = {3, 2, 2, 2, 2, 2};
    const std::vector<int> joinParent = {-1, 0, 1, 2, 0, 1};
    const std::vector<int> joinChild = {-1, 1, 0, 0, 0, 0};

    Table expected = bruteForce(tables, parent, joinParent, joinChild);
    std::vector<Table> working = tables;
    auto filtered = ParYanFilter(working, parent, 0, joinParent, joinChild, tableKeys);
    Table actual = ParYanJoin(filtered.first, parent, 0, joinParent, joinChild,
                              tableKeys, std::max(1, static_cast<int>(expected.size())));

    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    assert(actual == expected);
    std::cout << "ParYan memory-lifecycle test passed: rows=" << actual.size() << '\n';
    return 0;
}
