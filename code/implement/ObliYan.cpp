#include "../include/ObliYan.h"
#include "../include/BitonicSort.h"
#include "../include/ORcompact.h"
#include "../include/PrimitiveProfile.h"
#include "../include/sgx_profile.h"

#include <algorithm>
#ifndef SGX_ENCLAVE_BUILD
#include <chrono>
#endif
#include <climits>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>

#ifdef SGX_ENCLAVE_BUILD
#include "sgx_log.h"
#define cout sgx_cout
#define cerr sgx_cerr
#define endl sgx_endl
#endif

namespace
{
const int DO_DUMMY_VALUE = 100000007;
double g_lastObliYanUpFilterMs = 0.0;
double g_lastObliYanDownFilterMs = 0.0;
double g_lastObliYanJoinMs = 0.0;

struct ObliYanTwoWayTiming
{
    double augment = 0.0;
    double expand = 0.0;
    double multiNumber = 0.0;
    double sortAlign = 0.0;
    double result = 0.0;
};

// A zero-filled row is used as the materialized form of the paper's bottom tuple.
bool isBottom(const vector<int> &row)
{
    return all_of(row.begin(), row.end(), [](int v)
                      { return v == 0; });
    }

    vector<int> bottomRow(int width)
    {
    return vector<int>(width, 0);
}

// The parent array is the join tree; children[u] stores all outgoing tree edges.
vector<vector<int>> buildChildren(const vector<int> &parent)
{
        vector<vector<int>> children(parent.size());
        for (int i = 0; i < (int)parent.size(); ++i)
        {
            if (parent[i] != -1)
                children[parent[i]].push_back(i);
        }
    return children;
}

// Algorithm 17 uses bottom-up passes over every non-root node.
vector<int> bottomUpOrder(const vector<vector<int>> &children, int root)
{
        vector<int> order;
        queue<int> q;
        q.push(root);
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            order.push_back(u);
            for (int v : children[u])
                q.push(v);
        }
        reverse(order.begin(), order.end());
    return order;
}

// The top-down pass visits parents before their children.
vector<int> downOrder(const vector<vector<int>> &children, int root)
{
        vector<int> order;
        queue<int> q;
        q.push(root);
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            order.push_back(u);
            for (int v : children[u])
                q.push(v);
        }
    return order;
}

// BitonicSorter only sorts by table columns, so append a temporary flag column
// to keep bottom rows at the end while preserving the requested lexicographic keys.
Table bitonicSortWithBottomLast(Table table, vector<int> keyCols, bool ascending)
{
        if (table.empty())
            return {};

        int bottomFlagCol = (int)table[0].size();
        for (auto &row : table)
            row.push_back(isBottom(row) ? 1 : 0);

        vector<int> sortCols;
        sortCols.push_back(bottomFlagCol);
        sortCols.insert(sortCols.end(), keyCols.begin(), keyCols.end());

        BitonicSorter sorter(std::move(table), std::move(sortCols), ascending);
        sorter.Sorter();

        Table sorted = sorter.getSortedData();
        for (auto &row : sorted)
            row.pop_back();
    return sorted;
}

Table compactKeepingFirst(Table compacted, int keep)
{
    if (compacted.empty() || keep <= 0)
        return {};

    const int width = (int)compacted[0].size();
    vector<int> marker(compacted.size(), 0);
    for (int i = 0; i < (int)compacted.size(); ++i)
        marker[i] = isBottom(compacted[i]) ? 0 : 1;

    // Paper COMPACT: ORCompact stably moves rows whose marker is 1 to the front.
    ObliCompact().ORCompactSerial(compacted, marker, 0, (int)compacted.size());
    if ((int)compacted.size() > keep)
        compacted.resize(keep);
    while ((int)compacted.size() < keep)
        compacted.push_back(bottomRow(width));
    return compacted;
}

// ObliYan merges children in bottom-up order, while ParYanJoin exposes
// columns in BFS table order. Reorder only columns, not rows, before returning.
Table reorderColumnsByTableOrder(Table table,
                                  const vector<int> &currentTableOrder,
                                  const vector<int> &targetTableOrder,
                                  const vector<int> &tableWidths)
{
    if (table.empty())
        return table;

    vector<int> start(tableWidths.size(), -1);
    int pos = 0;
    for (int tableId : currentTableOrder)
    {
        start[tableId] = pos;
        pos += tableWidths[tableId];
    }

    for (int tableId : targetTableOrder)
    {
        int begin = start[tableId];
        int width = tableWidths[tableId];
        if (begin < 0 || begin + width > (int)table[0].size())
            return table;
    }

    for (auto &row : table)
    {
        vector<int> reordered;
        reordered.reserve(row.size());
        for (int tableId : targetTableOrder)
        {
            int begin = start[tableId];
            int width = tableWidths[tableId];
            reordered.insert(reordered.end(), row.begin() + begin, row.begin() + begin + width);
        }
        row = std::move(reordered);
    }
    return table;
}

bool debugObliYan()
{
#ifdef SGX_ENCLAVE_BUILD
    return false;
#else
    const char *env = getenv("RELAXED_JOIN_DEBUG");
    return env != nullptr && string(env) == "1";
#endif
}

void debugPrintTable(const string &name, const Table &table, int limit = 12)
{
    if (!debugObliYan())
        return;

    int realRows = 0;
    for (const auto &row : table)
    {
        if (!isBottom(row))
            ++realRows;
    }

#ifndef SGX_ENCLAVE_BUILD
    const char *limitEnv = getenv("RELAXED_JOIN_DEBUG_LIMIT");
    if (limitEnv != nullptr)
        limit = max(limit, atoi(limitEnv));
#endif

    cerr << name << " (" << table.size() << " rows, real=" << realRows << ")\n";
    for (int i = 0; i < (int)table.size() && i < limit; ++i)
    {
        cerr << "  [";
        for (int j = 0; j < (int)table[i].size(); ++j)
        {
            if (j)
                cerr << ", ";
            cerr << table[i][j];
        }
        cerr << "]\n";
    }
}

int countRealRows(const Table &table)
{
    int realRows = 0;
    for (const auto &row : table)
    {
        if (!isBottom(row))
            ++realRows;
    }
    return realRows;
}

#ifndef SGX_ENCLAVE_BUILD
double elapsedMs(chrono::high_resolution_clock::time_point start,
                 chrono::high_resolution_clock::time_point end)
{
    return chrono::duration<double, milli>(end - start).count();
}
#endif

void printObliYanTwoWayTiming(int p, int c, const ObliYanTwoWayTiming &timing)
{
#ifdef SGX_ENCLAVE_BUILD
    (void)p;
    (void)c;
    (void)timing;
#else
    cout << fixed << setprecision(1)
         << "  [ObliYan p=" << p << "->c=" << c << "]"
         << " augment=" << timing.augment << " ms"
         << "  expand=" << timing.expand << " ms"
         << "  multiNumber=" << timing.multiNumber << " ms"
         << "  sortAlign=" << timing.sortAlign << " ms"
         << "  result=" << timing.result << " ms\n"
         << defaultfloat << setprecision(6)
         << flush;
#endif
}

void printObliYanTwoWayTotal(const ObliYanTwoWayTiming &total)
{
#ifdef SGX_ENCLAVE_BUILD
    (void)total;
#else
    cout << fixed << setprecision(1)
         << "  [ObliYan TOTAL]"
         << " augment=" << total.augment << " ms"
         << "  expand=" << total.expand << " ms"
         << "  multiNumber=" << total.multiNumber << " ms"
         << "  sortAlign=" << total.sortAlign << " ms"
         << "  result=" << total.result << " ms\n"
         << defaultfloat << setprecision(6)
         << flush;
#endif
}

}

double getLastObliYanUpFilterMs()
{
    return g_lastObliYanUpFilterMs;
}

double getLastObliYanDownFilterMs()
{
    return g_lastObliYanDownFilterMs;
}

double getLastObliYanJoinMs()
{
    return g_lastObliYanJoinMs;
}

// Algorithm 10: keep each R tuple only if its join key appears in S.
Table SemiJoin(Table R, const Table &S, int joinColR, int joinColS)
{
    if (R.empty())
        return {};

    int rWidth = (int)R[0].size();
    int keepRows = (int)R.size();
    Table K;
    K.reserve(R.size() + S.size());

    for (const auto &s : S)
    {
        if (isBottom(s) || joinColS < 0 || joinColS >= (int)s.size())
        {
            K.push_back(bottomRow(rWidth + 2));
            continue;
        }

        vector<int> row;
        row.reserve(rWidth + 2);
        row.push_back(s[joinColS]);
        row.push_back(0); // S-tuples sort before R-tuples on the same key.
        row.insert(row.end(), rWidth, 0);
        K.push_back(row);
    }

    for (const auto &r : R)
    {
        if (isBottom(r) || joinColR < 0 || joinColR >= (int)r.size())
        {
            K.push_back(bottomRow(rWidth + 2));
            continue;
        }

        vector<int> row;
        row.reserve(rWidth + 2);
        row.push_back(r[joinColR]);
        row.push_back(1);
        row.insert(row.end(), r.begin(), r.end());
        K.push_back(row);
    }

    Table().swap(R);
    K = bitonicSortWithBottomLast(std::move(K), {0, 1}, true);

    int key = 0;
    bool hasKey = false;
    for (auto &t : K)
    {
        if (isBottom(t))
        {
            t = bottomRow(rWidth);
            continue;
        }

        if (t[1] == 0)
        {
            key = t[0];
            hasKey = true;
            t = bottomRow(rWidth);
        }
        else if (hasKey && t[0] == key)
        {
            t = vector<int>(t.begin() + 2, t.end());
        }
        else
        {
            t = bottomRow(rWidth);
        }
    }

    return compactKeepingFirst(std::move(K), keepRows);
}

// Algorithm 11: count tuples by key. The returned table is (key, count).
Table ReduceByKey(const Table &R, int keyCol)
{
    if (R.empty())
        return {};

    Table K = bitonicSortWithBottomLast(R, {keyCol}, true);

    int key = 0;
    int val = 0;
    bool hasKey = false;
    for (int i = 0; i < (int)K.size(); ++i)
    {
        const auto &row = K[i];
        if (isBottom(row) || keyCol < 0 || keyCol >= (int)row.size())
        {
            K[i] = bottomRow(2);
            continue;
        }

        if (!hasKey || row[keyCol] != key)
        {
            key = row[keyCol];
            val = 1;
            hasKey = true;
        }
        else
        {
            ++val;
        }

        bool groupEnds = (i + 1 == (int)K.size()) ||
                          isBottom(K[i + 1]) ||
                          K[i + 1][keyCol] != key;
        if (groupEnds)
            K[i] = {key, val};
        else
            K[i] = bottomRow(2);
    }

    return compactKeepingFirst(std::move(K), (int)R.size());
}

// Algorithm 12: attach the matching key count from S to each tuple in R.
Table Annotate(const Table &R, const Table &S, int keyColR, int keyColS)
{
    if (R.empty())
        return {};

    int rWidth = (int)R[0].size();
    Table K;
    K.reserve(R.size() + S.size());

    for (const auto &s : S)
    {
        if (isBottom(s) || keyColS < 0 || keyColS >= (int)s.size())
        {
            K.push_back(bottomRow(rWidth + 3));
            continue;
        }

        vector<int> row;
        row.reserve(rWidth + 3);
        row.push_back(s[keyColS]);
        row.push_back(0);
        row.push_back((int)s.size() > keyColS + 1 ? s[keyColS + 1] : 1);
        row.insert(row.end(), rWidth, 0);
        K.push_back(row);
    }

    for (const auto &r : R)
    {
        if (isBottom(r) || keyColR < 0 || keyColR >= (int)r.size())
        {
            K.push_back(bottomRow(rWidth + 3));
            continue;
        }

        vector<int> row;
        row.reserve(rWidth + 3);
        row.push_back(r[keyColR]);
        row.push_back(1);
        row.push_back(0);
        row.insert(row.end(), r.begin(), r.end());
        K.push_back(row);
    }

    K = bitonicSortWithBottomLast(std::move(K), {0, 1}, true);

    int key = 0;
    int val = 0;
    bool hasKey = false;
    for (auto &t : K)
    {
        if (isBottom(t))
        {
            t = bottomRow(rWidth + 1);
            continue;
        }

        if (t[1] == 0)
        {
            key = t[0];
            val = t[2];
            hasKey = true;
            t = bottomRow(rWidth + 1);
        }
        else if (hasKey && t[0] == key)
        {
            vector<int> row(t.begin() + 3, t.end());
            row.push_back(val);
            t = std::move(row);
        }
        else
        {
            t = bottomRow(rWidth + 1);
        }
    }

    return compactKeepingFirst(std::move(K), (int)R.size());
}

// Algorithm 14: reduce S by key, then annotate R with the reduced counts.
Table Augment(const Table &R, const Table &S, int joinColR, int joinColS)
{
    Table reduced = ReduceByKey(S, joinColS);
    return Annotate(R, reduced, joinColR, 0);
}

Table countByKeyList(const Table &R, int keyCol)
{
    Table reduced = ReduceByKey(R, keyCol);
    const int size = (int)reduced.size();
    return compactKeepingFirst(std::move(reduced), size);
}

Table MultiNumberInternal(Table R, int keyCol, const Table *copiesPerKey)
{
    if (R.empty())
        return {};

    vector<int> sortCols = {keyCol};
    for (int col = 0; col < (int)R[0].size(); ++col)
    {
        if (col != keyCol)
            sortCols.push_back(col);
    }
    R = bitonicSortWithBottomLast(std::move(R), std::move(sortCols), true);

    int key = 0;
    int val = 0;
    vector<int> lastTuple;
    bool hasKey = false;
    int countIdx = 0;
    int currentCopiesPerTuple = 0;
    int currentCopiesKey = 0;
    bool hasCopiesKey = false;
    for (auto &row : R)
    {
        if (isBottom(row) || keyCol < 0 || keyCol >= (int)row.size())
        {
            row = bottomRow((int)row.size() + 1);
            continue;
        }

        // Algorithm 13 numbers tuples after sorting by the join key. For bag
        // inputs, expanded copies of the same S tuple must reuse numbers
        // modulo the number of R tuples with this key so line 6 of Algorithm 16
        // aligns every R copy with every S tuple.
        if (copiesPerKey != nullptr && (!hasCopiesKey || currentCopiesKey != row[keyCol]))
        {
            currentCopiesPerTuple = 0;
            while (countIdx < (int)copiesPerKey->size() && isBottom((*copiesPerKey)[countIdx]))
                ++countIdx;
            while (countIdx < (int)copiesPerKey->size() &&
                   !isBottom((*copiesPerKey)[countIdx]) &&
                   (*copiesPerKey)[countIdx][0] < row[keyCol])
            {
                ++countIdx;
            }
            if (countIdx < (int)copiesPerKey->size() &&
                !isBottom((*copiesPerKey)[countIdx]) &&
                (*copiesPerKey)[countIdx][0] == row[keyCol])
            {
                currentCopiesPerTuple = (*copiesPerKey)[countIdx][1];
            }
            currentCopiesKey = row[keyCol];
            hasCopiesKey = true;
        }
        if (hasKey && row[keyCol] == key && row == lastTuple)
            ++val;
        else
            val = 1;
        key = row[keyCol];
        lastTuple = row;
        hasKey = true;

        row.push_back(currentCopiesPerTuple > 0 ? ((val - 1) % currentCopiesPerTuple) + 1 : val);
    }
    return R;
}

// Algorithm 13: number tuples within each key group after sorting by the key.
Table MultiNumber(Table R, int keyCol)
{
    return MultiNumberInternal(R, keyCol, nullptr);
}

// Algorithm 15: expand each tuple according to its trailing weight column.
Table Expand(Table R, int tau)
{
    PrimitiveProfileScope primitiveScope(PrimitiveExpand);
    if (tau <= 0)
        return {};
    if (R.empty())
        return {};

    const int posCol = 0;
    const int realCol = 1;
    const int infPos = INT_MAX - 1;
    int payloadWidth = max(0, (int)R[0].size() - 1);
    const int inputReal = debugObliYan() ? countRealRows(R) : 0;
    int pos = 2;
    int totalWeight = 0;
    Table K;
    K.reserve(R.size() + tau);
    for (const auto &row : R)
    {
        if (isBottom(row))
        {
            vector<int> entry(2 + payloadWidth, 0);
            entry[posCol] = infPos;
            entry[realCol] = 1;
            K.push_back(entry);
            continue;
        }

        int weight = row.empty() ? 1 : max(1, row.back());
        totalWeight = min(tau, totalWeight + weight);
        vector<int> trimmed(row.begin(), row.end() - 1);
        vector<int> entry;
        entry.reserve(2 + payloadWidth);
        entry.push_back(pos);
        entry.push_back(1);
        entry.insert(entry.end(), trimmed.begin(), trimmed.end());
        K.push_back(entry);
        pos += 2 * weight;
    }

    pos = 3;
    for (int i = 0; i < tau; ++i)
    {
        vector<int> entry(2 + payloadWidth, 0);
        entry[posCol] = pos;
        entry[realCol] = 0;
        K.push_back(entry);
        pos += 2;
    }

    Table().swap(R);
    BitonicSorter sorter(std::move(K), {posCol}, true);
    sorter.Sorter();
    K = sorter.getSortedData();

    vector<int> current = bottomRow(payloadWidth);
    int cnt = 0;
    for (auto &entry : K)
    {
        if (entry[posCol] == infPos)
        {
            entry = bottomRow(payloadWidth);
        }
        else if (entry[realCol] == 1)
        {
            current.assign(entry.begin() + 2, entry.end());
            entry = bottomRow(payloadWidth);
        }
        else if (cnt < tau && cnt < totalWeight)
        {
            entry = current;
            ++cnt;
        }
        else
        {
            entry = bottomRow(payloadWidth);
            ++cnt;
        }
    }

    const int beforeCompactReal = debugObliYan() ? countRealRows(K) : 0;
    Table expanded = compactKeepingFirst(std::move(K), tau);
    if (debugObliYan())
        cerr << "[Expand] inputReal=" << inputReal
             << " totalWeight=" << totalWeight
             << " beforeCompactReal=" << beforeCompactReal
             << " afterCompactReal=" << countRealRows(expanded) << "\n";
    return expanded;
}

// Algorithm 16: execute the paper's preparation steps, then materialize the
// two-way join result for the serial test implementation.
Table ObliYanTwoWayTimed(Table R,
                         Table S,
                         int joinColR,
                         int joinColS,
                         int tau,
                         ObliYanTwoWayTiming *timing)
{
    if (timing != nullptr)
        *timing = ObliYanTwoWayTiming{};
#ifndef SGX_ENCLAVE_BUILD
    auto t0 = chrono::high_resolution_clock::now();
#endif
    if (R.empty() || S.empty())
        return {};

    const int rWidth = (int)R[0].size();
    const int sWidth = (int)S[0].size();
    Table rCopiesPerKey = countByKeyList(R, joinColR);

    Table Rhat = Augment(R, S, joinColR, joinColS);
    debugPrintTable("Rhat", Rhat);
#ifndef SGX_ENCLAVE_BUILD
    auto augmentRDone = chrono::high_resolution_clock::now();
#endif
    // Algorithm 16 line 2. Consume each augmented table as soon as it is
    // available, so two augmented copies do not overlap with expansion
    // scratch space.
    Table Rtilde = Expand(std::move(Rhat), tau);
#ifndef SGX_ENCLAVE_BUILD
    auto expandRDone = chrono::high_resolution_clock::now();
#endif

    Table Shat = Augment(S, R, joinColS, joinColR);
    debugPrintTable("Shat", Shat);
#ifndef SGX_ENCLAVE_BUILD
    auto augmentSDone = chrono::high_resolution_clock::now();
#endif
    Table().swap(R);
    Table().swap(S);
    Table Stilde = Expand(std::move(Shat), tau);
#ifndef SGX_ENCLAVE_BUILD
    auto t2 = chrono::high_resolution_clock::now();
    const double augmentElapsed = elapsedMs(t0, augmentRDone) +
                                  elapsedMs(expandRDone, augmentSDone);
    const double expandElapsed = elapsedMs(augmentRDone, expandRDone) +
                                 elapsedMs(augmentSDone, t2);
#endif
    debugPrintTable("Rtilde", Rtilde);
    debugPrintTable("Stilde after Expand", Stilde);

    // Algorithm 16 lines 3-4: number S within each join-key group, then sort by
    // the join key and the generated number.
    Stilde = MultiNumberInternal(std::move(Stilde), joinColS, &rCopiesPerKey);
    Table().swap(rCopiesPerKey);
#ifndef SGX_ENCLAVE_BUILD
    auto t3 = chrono::high_resolution_clock::now();
#endif
    debugPrintTable("Stilde after MultiNumber", Stilde);
    if (!Stilde.empty())
    {
        int numCol = (int)Stilde[0].size() - 1;
        vector<int> sortCols = {joinColS, numCol};
        for (int col = 0; col < numCol; ++col)
        {
            if (col != joinColS)
                sortCols.push_back(col);
        }
        Stilde = bitonicSortWithBottomLast(std::move(Stilde), std::move(sortCols), true);
    }
#ifndef SGX_ENCLAVE_BUILD
    auto t4 = chrono::high_resolution_clock::now();
#endif
    debugPrintTable("Stilde after Sort", Stilde);

    Rtilde.resize(tau, bottomRow(rWidth));
    for (int i = 0; i < tau; ++i)
    {
        if (i >= (int)Rtilde.size() || i >= (int)Stilde.size() ||
            isBottom(Rtilde[i]) || isBottom(Stilde[i]))
        {
            Rtilde[i] = bottomRow(rWidth + sWidth);
            continue;
        }

        Rtilde[i].insert(Rtilde[i].end(), Stilde[i].begin(), Stilde[i].begin() + sWidth);
    }
    Table().swap(Stilde);
#ifndef SGX_ENCLAVE_BUILD
    auto t5 = chrono::high_resolution_clock::now();

    if (timing != nullptr)
    {
        timing->augment = augmentElapsed;
        timing->expand = expandElapsed;
        timing->multiNumber = elapsedMs(t2, t3);
        timing->sortAlign = elapsedMs(t3, t4);
        timing->result = elapsedMs(t4, t5);
    }
#endif

    return Rtilde;
}

Table ObliYanTwoWay(Table R, Table S, int joinColR, int joinColS, int tau)
{
    return ObliYanTwoWayTimed(std::move(R), std::move(S), joinColR, joinColS, tau, nullptr);
}

// Algorithm 17, lines 6-14: bottom-up semi-join, top-down semi-join,
// then bottom-up ObliYanTwoWay joins until the root holds the final result.
Table ObliYan(vector<Table> tables,
                  const vector<int> &parent,
                  int root,
                  const vector<int> &joinColInParent,
                  const vector<int> &joinColInChild,
                  int tau)
{
    g_lastObliYanUpFilterMs = 0.0;
    g_lastObliYanDownFilterMs = 0.0;
    g_lastObliYanJoinMs = 0.0;
    double sgxTotalStart = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
    auto totalStart = chrono::high_resolution_clock::now();
    cout << "  [ObliYan] start: tables=" << tables.size()
         << " tau=" << tau << "\n"
         << flush;
#endif

    vector<int> tableWidths(tables.size(), 0);
    for (int i = 0; i < (int)tables.size(); ++i)
    {
        if (!tables[i].empty())
            tableWidths[i] = (int)tables[i][0].size();
    }

    vector<vector<int>> children = buildChildren(parent);
    vector<int> bu = bottomUpOrder(children, root);
    vector<int> bfs = downOrder(children, root);
    vector<vector<int>> tableOrder(tables.size());
    for (int i = 0; i < (int)tables.size(); ++i)
        tableOrder[i] = {i};

    setPrimitiveProfilePhase(PrimitivePhaseObliYanUpFilter);
    for (int u : bu)
    {
        if (u == root)
            continue;
        int p = parent[u];
        tables[p] = SemiJoin(std::move(tables[p]), tables[u], joinColInParent[u], joinColInChild[u]);
    }
    double sgxBottomUpEnd = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
    auto bottomUpEnd = chrono::high_resolution_clock::now();
    cout << "  [ObliYan] bottom-up semijoin done. elapsed="
         << elapsedMs(totalStart, bottomUpEnd) << " ms\n"
         << flush;
#endif

    setPrimitiveProfilePhase(PrimitivePhaseObliYanDownFilter);
    vector<int> td = downOrder(children, root);
    for (int u : td)
    {
        if (children[u].empty())
            continue;
        for (int v : children[u])
            tables[v] = SemiJoin(std::move(tables[v]), tables[u], joinColInChild[v], joinColInParent[v]);
    }
    double sgxObliYanDownEnd = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
    auto downEnd = chrono::high_resolution_clock::now();
    cout << "  [ObliYan] top-down semijoin done. stage="
         << elapsedMs(bottomUpEnd, downEnd) << " ms  elapsed="
         << elapsedMs(totalStart, downEnd) << " ms\n"
         << flush;
#endif

    setPrimitiveProfilePhase(PrimitivePhaseObliYanJoin);
    ObliYanTwoWayTiming obliYanTotal;
    int edgeIndex = 0;
    int edgeCount = 0;
    for (int u : bu)
    {
        if (u != root)
            ++edgeCount;
    }
    for (int u : bu)
    {
        if (u == root)
            continue;
        int p = parent[u];
        ++edgeIndex;
#ifndef SGX_ENCLAVE_BUILD
        auto edgeStart = chrono::high_resolution_clock::now();
        cout << "  [ObliYan] ObliYanTwoWay edge " << edgeIndex << "/" << edgeCount
             << " p=" << p << " c=" << u
             << " parentRows=" << tables[p].size()
             << " childRows=" << tables[u].size()
             << " parentReal=" << countRealRows(tables[p])
             << " childReal=" << countRealRows(tables[u])
             << "\n"
             << flush;
#endif
        if (debugObliYan())
            cerr << "[ObliYan edge " << p << "->" << u << "] before: parentReal="
                 << countRealRows(tables[p]) << " childReal=" << countRealRows(tables[u]) << "\n";
        ObliYanTwoWayTiming edgeTiming;
        int parentInputRows = (int)tables[p].size();
        int childInputRows = (int)tables[u].size();
        tables[p] = ObliYanTwoWayTimed(std::move(tables[p]), std::move(tables[u]),
                                      joinColInParent[u], joinColInChild[u], tau, &edgeTiming);
        if (sgxProfileEnabled())
        {
            char buf[240];
            snprintf(buf, sizeof(buf),
                     "  [ObliYan edge p=%d c=%d] inputRows=%d,%d tau=%d resultRows=%d cols=%d",
                     p, u, parentInputRows, childInputRows, tau,
                     (int)tables[p].size(), tables[p].empty() ? 0 : (int)tables[p][0].size());
            sgxProfilePrint(buf);
        }
        obliYanTotal.augment += edgeTiming.augment;
        obliYanTotal.expand += edgeTiming.expand;
        obliYanTotal.multiNumber += edgeTiming.multiNumber;
        obliYanTotal.sortAlign += edgeTiming.sortAlign;
        obliYanTotal.result += edgeTiming.result;
        printObliYanTwoWayTiming(p, u, edgeTiming);
#ifndef SGX_ENCLAVE_BUILD
        auto edgeEnd = chrono::high_resolution_clock::now();
        cout << "  [ObliYan] ObliYanTwoWay edge " << edgeIndex << "/" << edgeCount
             << " done. resultRows=" << tables[p].size()
             << " resultReal=" << countRealRows(tables[p])
             << " stage=" << elapsedMs(edgeStart, edgeEnd) << " ms  elapsed="
             << elapsedMs(totalStart, edgeEnd) << " ms\n"
             << flush;
#endif
        if (debugObliYan())
            cerr << "[ObliYan edge " << p << "->" << u << "] after: parentReal="
                 << countRealRows(tables[p]) << "\n";
        tableOrder[p].insert(tableOrder[p].end(), tableOrder[u].begin(), tableOrder[u].end());
    }
    double sgxJoinEnd = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
    auto obliYanTwoWayEnd = chrono::high_resolution_clock::now();
#endif
    printObliYanTwoWayTotal(obliYanTotal);

    Table result = reorderColumnsByTableOrder(std::move(tables[root]), tableOrder[root], bfs, tableWidths);
    size_t write = 0;
    for (size_t read = 0; read < result.size(); ++read)
    {
        if (!isBottom(result[read]))
        {
            if (write != read)
                result[write] = std::move(result[read]);
            ++write;
        }
    }
    result.resize(write);
    if (tau > 0 && (int)result.size() < tau)
    {
        int width = 0;
        if (!result.empty())
            width = (int)result[0].size();
        else
            for (int tableWidth : tableWidths)
                width += tableWidth;
        result.resize(tau, vector<int>(width, DO_DUMMY_VALUE));
    }
    double sgxCompactEnd = sgxProfileNowMs();
    setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
    if (sgxCompactEnd > 0.0 && sgxTotalStart > 0.0)
    {
        g_lastObliYanUpFilterMs = sgxBottomUpEnd - sgxTotalStart;
        g_lastObliYanDownFilterMs = sgxObliYanDownEnd - sgxBottomUpEnd;
        g_lastObliYanJoinMs = sgxCompactEnd - sgxObliYanDownEnd;
    }
#ifndef SGX_ENCLAVE_BUILD
    auto compactEnd = chrono::high_resolution_clock::now();
    cout << "  [ObliYan] final compact done. resultRows=" << result.size()
         << " stage=" << elapsedMs(obliYanTwoWayEnd, compactEnd)
         << " ms  elapsed=" << elapsedMs(totalStart, compactEnd) << " ms\n"
         << flush;

    double msBottomUpSemi = elapsedMs(totalStart, bottomUpEnd);
    double msObliYanDownSemi = elapsedMs(bottomUpEnd, downEnd);
    double msObliYanTwoWay = elapsedMs(downEnd, obliYanTwoWayEnd);
    double msCompact = elapsedMs(obliYanTwoWayEnd, compactEnd);
    double msTotal = elapsedMs(totalStart, compactEnd);
    g_lastObliYanUpFilterMs = msBottomUpSemi;
    g_lastObliYanDownFilterMs = msObliYanDownSemi;
    g_lastObliYanJoinMs = msObliYanTwoWay + msCompact;
    cout << "  [ObliYan detail] bottomUpSemi=" << msBottomUpSemi
         << " ms  ObliYanDownSemi=" << msObliYanDownSemi
         << " ms  ObliYanTwoWay=" << msObliYanTwoWay
         << " ms  compact=" << msCompact
         << " ms  total=" << msTotal << " ms\n";
#endif

    return result;
}
