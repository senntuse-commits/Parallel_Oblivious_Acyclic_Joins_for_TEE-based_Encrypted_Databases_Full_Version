#include "../include/BottomupSemiJoin.h"
#include "../include/JFYanDown.h"
#include "../include/ObliViatorExpand.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <omp.h>

static Table reference(const vector<Table> &tables, const vector<int> &parent,
                       const vector<int> &jp, const vector<int> &jc)
{
    Table result;
    vector<int> selection(tables.size(), 0);
    std::function<void(int)> visit = [&](int r)
    {
        if (r == static_cast<int>(tables.size()))
        {
            for (int c = 0; c < r; ++c)
                if (parent[c] >= 0 && tables[c][selection[c]][jc[c]] !=
                    tables[parent[c]][selection[parent[c]]][jp[c]])
                    return;
            vector<int> row;
            for (int node = 0; node < r; ++node)
                row.insert(row.end(), tables[node][selection[node]].begin(),
                           tables[node][selection[node]].end());
            result.push_back(std::move(row));
            return;
        }
        for (int i = 0; i < static_cast<int>(tables[r].size()); ++i)
        {
            selection[r] = i;
            visit(r + 1);
        }
    };
    visit(0);
    std::sort(result.begin(), result.end());
    return result;
}

static void check(const vector<Table> &tables, const vector<int> &parent,
                  const vector<int> &jp, const vector<int> &jc, int root = 0)
{
    const Table expected = reference(tables, parent, jp, jc);
    vector<int> widths(tables.size(), 2);
    for (int i = 0; i < static_cast<int>(tables.size()); ++i)
        if (!tables[i].empty())
            widths[i] = static_cast<int>(tables[i][0].size());
    for (int padding : {0, 1, 5})
    {
        const int target = static_cast<int>(expected.size()) + padding;
        auto working = tables;
        working[root] = bottomUpSemiJoin(working, parent, root, jp, jc, target, widths[root]);
        assert(working[root].size() == static_cast<size_t>(target));
        Table actual = JFYanDown(working, parent, root, jp, jc, widths);
        const auto &validity = getLastJFYanOutputValidity();
        assert(actual.size() == static_cast<size_t>(target));
        assert(validity.size() == actual.size());
        Table real;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            assert(validity[i] == 0 || validity[i] == 1);
            if (validity[i])
                real.push_back(actual[i]);
        }
        std::sort(real.begin(), real.end());
        if (real != expected)
        {
            std::cerr << "Join regression failed: relations=" << tables.size()
                      << " target=" << target << " expected=" << expected.size()
                      << " actual=" << real.size() << '\n';
            for (const auto &row : expected)
            {
                std::cerr << "expected ";
                for (int v : row)
                    std::cerr << v << ' ';
                std::cerr << '\n';
            }
            for (const auto &row : real)
            {
                std::cerr << "actual ";
                for (int v : row)
                    std::cerr << v << ' ';
                std::cerr << '\n';
            }
            std::abort();
        }
    }
}

int main(int argc, char **argv)
{
    omp_set_dynamic(0);
    const int threads = argc > 1 ? std::atoi(argv[1]) : 2;
    assert(threads > 0 && threads <= 8);
    omp_set_num_threads(threads);
    std::mt19937 rng(91827);
    for (int trial = 0; trial < 80; ++trial)
    {
        int n = 1 + rng() % 9;
        Table rows(n, vector<int>(2));
        vector<int> counts(n);
        Table expected;
        for (int i = 0; i < n; ++i)
        {
            rows[i] = {i % 2 == 0 ? INT_MAX : INT_MIN, i};
            counts[i] = rng() % 4;
            for (int k = 0; k < counts[i]; ++k)
                expected.push_back(rows[i]);
        }
        assert(ObliViatorExpand(rows, counts, static_cast<int>(expected.size())).getResult() == expected);
    }
    assert(ObliViatorExpand({{7}, {8}, {9}}, {0, 0, 0}, 0).getResult().empty());
    assert((ObliViatorExpand({{7}, {8}, {9}}, {0, 1, 0}, 1).getResult() == Table{{8}}));

    // An ancestor right sibling gives inherited lambda=2 on the long branch.
    check({{{1,2}}, {{1,3}}, {{3,4}}, {{4,40},{4,41}}, {{2,20},{2,21}}},
          {-1,0,1,2,0}, {-1,0,1,1,1}, {-1,0,0,0,0});
    check({{{0,10},{1,11},{2,12},{3,13}}, {{2,20}}}, {-1,0}, {-1,0}, {-1,0});
    check({{{INT_MIN,10},{INT_MAX,11},{0,12}}, {{INT_MIN,20},{INT_MAX,21},{0,22}}},
          {-1,0}, {-1,0}, {-1,0});
    check({{{1,10}}, {{2,20}}}, {-1,0}, {-1,0}, {-1,0});
    check({{}, {{1,20}}}, {-1,0}, {-1,0}, {-1,0});
    check({{{1,10}}, {}}, {-1,0}, {-1,0}, {-1,0});
    check({{{1,10}}, {}, {{1,20}}}, {-1,0,1}, {-1,0,0}, {-1,0,0});
    check({{{1,10},{2,20}}}, {-1}, {-1}, {-1});
    check({{}}, {-1}, {-1}, {-1});
    check({{{1,10},{2,20}}, {{1,30},{1,40}}}, {1,-1}, {0,-1}, {0,-1}, 1);
    for (int trial = 0; trial < 50; ++trial)
    {
        const int n = 2 + rng() % 5;
        vector<Table> tables(n);
        vector<int> parent(n, -1), jp(n, 1), jc(n, 0);
        for (int node = 0; node < n; ++node)
        {
            if (node > 0)
                parent[node] = rng() % node;
            const int size = 1 + rng() % 3;
            for (int row = 0; row < size; ++row)
                tables[node].push_back({static_cast<int>(rng() % 3), static_cast<int>(rng() % 3)});
        }
        check(tables, parent, jp, jc);
    }
    bool rejected = false;
    try
    {
        vector<Table> tables = {{{1,2},{1,3}}};
        bottomUpSemiJoin(tables, {-1}, 0, {-1}, {-1}, 1, 2);
    }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    std::cout << "JFYan fixed-target correctness regressions passed (threads=" << threads << ")\n";
}
