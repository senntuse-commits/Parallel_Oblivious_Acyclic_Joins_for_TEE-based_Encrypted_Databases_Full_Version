#include "../include/ObliYan.h"
#include "../App/util.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
Table loadRequired(const std::string &dir, const char *name, int cols)
{
    Table table = loadData(dir + "/" + name);
    if (table.empty())
        throw std::runtime_error(std::string("empty table: ") + name);
    for (const auto &row : table)
        if ((int)row.size() != cols)
            throw std::runtime_error(std::string("bad width: ") + name);
    return table;
}

std::uint64_t rowHash(const std::vector<int> &row)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (int value : row)
    {
        hash ^= static_cast<std::uint32_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void printSignature(const Table &table)
{
    std::uint64_t sum = 0;
    std::uint64_t mixed = 0;
    for (const auto &row : table)
    {
        const std::uint64_t hash = rowHash(row);
        sum += hash;
        mixed += hash * (hash | 1ULL);
    }
    std::cout << "rows=" << table.size()
              << " cols=" << (table.empty() ? 0 : table[0].size())
              << " hash_sum=" << sum
              << " hash_mixed=" << mixed << '\n';
}

int runSmallCorrectness()
{
    std::vector<Table> tables = {
        {{1, 10}, {1, 11}, {2, 20}},
        {{1, 100}, {1, 101}, {2, 200}},
        {{1, 300}, {2, 400}, {2, 401}},
        {{100, 1000}, {100, 1001}, {101, 1010}, {200, 2000}},
    };
    const std::vector<int> parent = {-1, 0, 0, 1};
    const std::vector<int> joinParent = {-1, 0, 0, 1};
    const std::vector<int> joinChild = {-1, 0, 0, 0};

    Table expected;
    for (const auto &r0 : tables[0])
        for (const auto &r1 : tables[1])
            if (r0[0] == r1[0])
                for (const auto &r2 : tables[2])
                    if (r0[0] == r2[0])
                        for (const auto &r3 : tables[3])
                            if (r1[1] == r3[0])
                            {
                                std::vector<int> row;
                                row.insert(row.end(), r0.begin(), r0.end());
                                row.insert(row.end(), r1.begin(), r1.end());
                                row.insert(row.end(), r2.begin(), r2.end());
                                row.insert(row.end(), r3.begin(), r3.end());
                                expected.push_back(std::move(row));
                            }

#ifdef BASELINE_COPY_INPUT
    Table actual = ObliYan(tables, parent, 0, joinParent, joinChild, (int)expected.size());
#else
    Table actual = ObliYan(std::move(tables), parent, 0, joinParent, joinChild, (int)expected.size());
#endif
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual != expected)
    {
        std::cerr << "small correctness failed: actual=" << actual.size()
                  << " expected=" << expected.size() << '\n';
        return 1;
    }
    std::cout << "small correctness passed ";
    printSignature(actual);
    return 0;
}

int runQ18(const std::string &dir)
{
    std::vector<Table> tables = {
        loadRequired(dir, "R1_catalog_sales.tbl", 4),
        loadRequired(dir, "R2_date_dim.tbl", 2),
        loadRequired(dir, "R3_item.tbl", 2),
        loadRequired(dir, "R4_cd1.tbl", 2),
        loadRequired(dir, "R5_customer.tbl", 3),
        loadRequired(dir, "R6_cd2.tbl", 2),
        loadRequired(dir, "R7_customer_address.tbl", 2),
    };
    std::ifstream expectedFile(dir + "/expected.txt");
    int expectedRows = 0;
    expectedFile >> expectedRows;
    if (expectedRows <= 0)
        throw std::runtime_error("bad expected.txt");

    const std::vector<int> parent = {-1, 0, 0, 0, 0, 4, 4};
    const std::vector<int> joinParent = {-1, 0, 1, 2, 3, 1, 2};
    const std::vector<int> joinChild = {-1, 0, 0, 0, 0, 0, 0};
#ifdef BASELINE_COPY_INPUT
    Table result = ObliYan(tables, parent, 0, joinParent, joinChild, expectedRows);
#else
    Table result = ObliYan(std::move(tables), parent, 0, joinParent, joinChild, expectedRows);
#endif
    if ((int)result.size() != expectedRows)
    {
        std::cerr << "Q18 row-count failed: actual=" << result.size()
                  << " expected=" << expectedRows << '\n';
        return 1;
    }
    printSignature(result);
    return 0;
}
}

int main(int argc, char **argv)
{
    try
    {
        if (argc == 1)
            return runSmallCorrectness();
        if (argc == 2)
            return runQ18(argv[1]);
        std::cerr << "usage: obliyan_test [q18_dir]\n";
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
