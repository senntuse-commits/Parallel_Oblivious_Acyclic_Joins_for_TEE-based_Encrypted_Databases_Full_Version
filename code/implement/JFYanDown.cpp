#include "../include/JFYanDown.h"
#include "../include/Aggtree.h"
#include "../include/ORcompact.h"
#include "../include/DupAggtree.h"
#include "../include/ParallelBitonicSort.h"
#include "../include/PrimitiveProfile.h"
#include "../include/MemoryProfile.h"
#include "../include/sgx_profile.h"

// JFYanDown.cpp materializes filtered acyclic join results from the root toward leaves.
// The implementation keeps profiling hooks close to each primitive so enclave
// timing output can explain where top-down expansion and compaction time is spent.
#ifndef SGX_ENCLAVE_BUILD
#include <chrono>
#endif
#include <climits>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>
#include <omp.h>
#ifndef SGX_ENCLAVE_BUILD
using namespace std::chrono;
#endif

static vector<int> g_lastJFYanOutputValidity;
const vector<int> &getLastJFYanOutputValidity() { return g_lastJFYanOutputValidity; }
static int selectValue(bool condition, int yes, int no)
{
    const uint32_t mask = 0u - static_cast<uint32_t>(condition);
    return static_cast<int>((static_cast<uint32_t>(yes) & mask) |
                            (static_cast<uint32_t>(no) & ~mask));
}

static double g_lastJFYanDownParallelMs = 0.0;

double getLastJFYanDownParallelMs()
{
    return g_lastJFYanDownParallelMs;
}

static void sgxJFYanDownLog(const char *fmt, int a, int b, double x, double y, double z)
{
    if (!sgxProfileEnabled())
        return;
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, a, b, x, y, z);
    sgxProfilePrint(buf);
}

static int chooseJFYanDownWorkerCount(const vector<Table> &tables,
                                    const vector<int> &children,
                                    int parentRows,
                                    int maxThreads)
{
    int childCount = (int)children.size();
    if (childCount <= 1 || maxThreads <= 16)
        return 1;

    long long largestEdgeRows = 0;
    long long totalEdgeRows = 0;
    for (int c : children)
    {
        long long edgeRows = (long long)parentRows + (long long)tables[c].size();
        largestEdgeRows = std::max(largestEdgeRows, edgeRows);
        totalEdgeRows += edgeRows;
    }

    int workerCap = maxThreads >= 64 ? 4 : 2;
    if (largestEdgeRows >= 6000000LL || totalEdgeRows >= 30000000LL)
        workerCap = 2;
    return std::max(1, std::min(childCount, workerCap));
}

static int splitInnerThreadCount(int maxThreads, int workers)
{
    if (workers <= 1)
        return std::max(1, maxThreads);
    // Reserve one TCS per outer worker. Without this, 64 requested threads
    // with 4 outer workers can require 4 + 4*16 TCS and force serialization.
    return std::max(1, (maxThreads - workers) / workers);
}

struct FlatTable
{
    int rows = 0;
    int cols = 0;
    vector<int> data;

    FlatTable() = default;
    FlatTable(int r, int c) : rows(r), cols(c), data((size_t)r * c, 0) {}

    inline int &at(int r, int c)
    {
        return data[(size_t)r * cols + c];
    }

    inline const int &at(int r, int c) const
    {
        return data[(size_t)r * cols + c];
    }
    inline int get(int r, int c) const { return at(r, c); }
    inline void set(int r, int c, int value) { at(r, c) = value; }
    inline void resizeRows(int count) { data.resize((size_t)count * cols, 0); rows = count; }
};

static unsigned long long flatTableStorageBytes(const vector<FlatTable> &tables)
{
    unsigned long long bytes =
        static_cast<unsigned long long>(tables.capacity()) * sizeof(FlatTable);
    for (const FlatTable &table : tables)
        bytes += static_cast<unsigned long long>(table.data.capacity()) * sizeof(int);
    return bytes;
}

static int chooseFinalSortWorkerCount(const vector<FlatTable> &tempResult,
                                      int root,
                                      int finalSortCount,
                                      int maxThreads)
{
    if (finalSortCount <= 1 || maxThreads <= 16)
        return 1;

    int maxRows = 0;
    long long totalRows = 0;
    for (int c = 0; c < (int)tempResult.size(); ++c)
    {
        if (c == root || tempResult[c].rows <= 0)
            continue;
        maxRows = std::max(maxRows, tempResult[c].rows);
        totalRows += tempResult[c].rows;
    }

    int workerCap = maxThreads >= 64 ? 4 : 2;
    if (maxRows >= 6000000 || totalRows >= 30000000LL)
        workerCap = 2;
    return std::max(1, std::min(finalSortCount, workerCap));
}

struct PackedFlatTable
{
    int rows = 0;
    int cols = 0;
    int wordsPerRow = 0;
    vector<uint64_t> data;

    PackedFlatTable() = default;
    PackedFlatTable(int r, int c)
        : rows(r), cols(c), wordsPerRow((c + 1) / 2), data((size_t)r * wordsPerRow, 0) {}

    inline int get(int r, int c) const
    {
        uint64_t word = data[(size_t)r * wordsPerRow + (c >> 1)];
        return (c & 1) ? (int)(uint32_t)(word >> 32) : (int)(uint32_t)word;
    }

    inline void set(int r, int c, int value)
    {
        uint64_t &word = data[(size_t)r * wordsPerRow + (c >> 1)];
        uint64_t v = (uint32_t)value;
        if (c & 1)
            word = (word & 0x00000000ffffffffULL) | (v << 32);
        else
            word = (word & 0xffffffff00000000ULL) | v;
    }

    inline void resizeRows(int newRows)
    {
        data.resize((size_t)newRows * wordsPerRow, 0);
        rows = newRows;
    }
};

static FlatTable subProcessFlat(const Table &parent, int joinColInParent, Table &child, int joinColInChild,
                                const vector<int> &childRank, const vector<int> &parentIdx,
                                const vector<int> &scFlat, const vector<int> &pcFlat,
                                int siblingCount, int c, bool isLeafNode, int childNum, int keyCols,
                                int workerThreads = 0, int childArity = -1);

static void flatSwapRows(FlatTable &D, int i, int j, bool b)
{
    unsigned int mask = 0u - (unsigned int)(b & (i != j));
    uint64_t mask64 = 0ULL - (uint64_t)(b & (i != j));
    int *ri = D.data.data() + (size_t)i * D.cols;
    int *rj = D.data.data() + (size_t)j * D.cols;
    int k = 0;
    for (; k + 1 < D.cols; k += 2)
    {
        uint64_t x = ((uint64_t)(uint32_t)ri[k]) | ((uint64_t)(uint32_t)ri[k + 1] << 32);
        uint64_t y = ((uint64_t)(uint32_t)rj[k]) | ((uint64_t)(uint32_t)rj[k + 1] << 32);
        uint64_t diff = (x ^ y) & mask64;
        x ^= diff;
        y ^= diff;
        ri[k] = (int)(uint32_t)x;
        ri[k + 1] = (int)(uint32_t)(x >> 32);
        rj[k] = (int)(uint32_t)y;
        rj[k + 1] = (int)(uint32_t)(y >> 32);
    }
    for (; k < D.cols; ++k)
    {
        unsigned int x = (unsigned int)ri[k];
        unsigned int y = (unsigned int)rj[k];
        unsigned int diff = (x ^ y) & mask;
        ri[k] = (int)(x ^ diff);
        rj[k] = (int)(y ^ diff);
    }
}

static inline void swapUint64(uint64_t &a, uint64_t &b, bool cond)
{
    uint64_t mask = 0ULL - (uint64_t)cond;
    uint64_t diff = (a ^ b) & mask;
    a ^= diff;
    b ^= diff;
}

static inline void swapUint128(unsigned __int128 &a, unsigned __int128 &b, bool cond)
{
    unsigned __int128 mask = (unsigned __int128)0 - (unsigned __int128)cond;
    unsigned __int128 diff = (a ^ b) & mask;
    a ^= diff;
    b ^= diff;
}

static void packedSwapRows(PackedFlatTable &D, int i, int j, bool b)
{
    uint64_t mask = 0ULL - (uint64_t)(b & (i != j));
    uint64_t *ri = D.data.data() + (size_t)i * D.wordsPerRow;
    uint64_t *rj = D.data.data() + (size_t)j * D.wordsPerRow;
    for (int k = 0; k < D.wordsPerRow; ++k)
    {
        uint64_t diff = (ri[k] ^ rj[k]) & mask;
        ri[k] ^= diff;
        rj[k] ^= diff;
    }
}

static void flatOffCompact(FlatTable &D, const vector<int> &M, int left, int right, int z)
{
    int n = right - left;
    if (n == 2)
    {
        flatSwapRows(D, left, left + 1, ((1 - M[left]) * M[left + 1]) ^ z);
    }
    else if (n > 2)
    {
        int mid = left + n / 2;
        int m = 0;
        for (int i = left; i < mid; ++i)
            m += M[i];

        flatOffCompact(D, M, left, mid, z % (n / 2));
        flatOffCompact(D, M, mid, right, (z + m) % (n / 2));

        bool s = ((z % (n / 2) + m) >= (n / 2)) ^ (z >= (n / 2));
        for (int i = 0; i < n / 2; ++i)
        {
            bool b = s ^ (i >= ((z + m) % (n / 2)));
            flatSwapRows(D, left + i, left + i + n / 2, b);
        }
    }
}

static void flatORCompact(FlatTable &D, const vector<int> &M, int left, int right)
{
    int n = right - left;
    if (n == 0)
        return;

    int n1 = 1;
    while ((n1 << 1) <= n)
        n1 <<= 1;
    int n2 = n - n1;
    int m = 0;
    for (int i = left; i < left + n2; ++i)
        m += M[i];

    flatORCompact(D, M, left, left + n2);
    flatOffCompact(D, M, left + n2, right, (n1 - n2 + m) % n1);

    for (int i = 0; i < n2; ++i)
        flatSwapRows(D, left + i, left + n1 + i, i >= m);
}

static void packedOffCompact(PackedFlatTable &D, const vector<int> &M, int left, int right, int z)
{
    int n = right - left;
    if (n == 2)
    {
        packedSwapRows(D, left, left + 1, ((1 - M[left]) * M[left + 1]) ^ z);
    }
    else if (n > 2)
    {
        int mid = left + n / 2;
        int m = 0;
        for (int i = left; i < mid; ++i)
            m += M[i];

        packedOffCompact(D, M, left, mid, z % (n / 2));
        packedOffCompact(D, M, mid, right, (z + m) % (n / 2));

        bool s = ((z % (n / 2) + m) >= (n / 2)) ^ (z >= (n / 2));
        for (int i = 0; i < n / 2; ++i)
        {
            bool b = s ^ (i >= ((z + m) % (n / 2)));
            packedSwapRows(D, left + i, left + i + n / 2, b);
        }
    }
}

static void packedORCompact(PackedFlatTable &D, const vector<int> &M, int left, int right)
{
    int n = right - left;
    if (n == 0)
        return;

    int n1 = 1;
    while ((n1 << 1) <= n)
        n1 <<= 1;
    int n2 = n - n1;
    int m = 0;
    for (int i = left; i < left + n2; ++i)
        m += M[i];

    packedORCompact(D, M, left, left + n2);
    packedOffCompact(D, M, left + n2, right, (n1 - n2 + m) % n1);

    for (int i = 0; i < n2; ++i)
        packedSwapRows(D, left + i, left + n1 + i, i >= m);
}

static vector<int> prefixSumMarksPar(const vector<int> &M, int threads)
{
    int n = (int)M.size();
    vector<int> S(n + 1, 0);
    if (n == 0)
        return S;

    int blocks = std::max(1, std::min(threads, n));
    int chunk = (n + blocks - 1) / blocks;
    vector<int> blockSum(blocks, 0);

#pragma omp parallel for schedule(static) num_threads(threads)
    for (int b = 0; b < blocks; ++b)
    {
        int begin = b * chunk;
        int end = std::min(n, begin + chunk);
        int sum = 0;
        for (int i = begin; i < end; ++i)
            sum += M[i];
        blockSum[b] = sum;
    }

    int offset = 0;
    for (int b = 0; b < blocks; ++b)
    {
        int sum = blockSum[b];
        blockSum[b] = offset;
        offset += sum;
    }

#pragma omp parallel for schedule(static) num_threads(threads)
    for (int b = 0; b < blocks; ++b)
    {
        int begin = b * chunk;
        int end = std::min(n, begin + chunk);
        int running = blockSum[b];
        for (int i = begin; i < end; ++i)
        {
            running += M[i];
            S[i + 1] = running;
        }
    }
    return S;
}

static void packedOROCPARTask(PackedFlatTable &D, const vector<int> &S, int left, int right, int z)
{
    int n = right - left;
    if (n == 2)
    {
        int m0 = S[left + 1] - S[left];
        int m1 = S[left + 2] - S[left + 1];
        packedSwapRows(D, left, left + 1, ((1 - m0) * m1) ^ z);
    }
    else if (n > 2)
    {
        int half = n / 2;
        int mid = left + half;
        int m = S[mid] - S[left];

#pragma omp task shared(D, S) firstprivate(left, mid, z, half) if (n >= 8192)
        packedOROCPARTask(D, S, left, mid, z % half);
#pragma omp task shared(D, S) firstprivate(mid, right, z, m, half) if (n >= 8192)
        packedOROCPARTask(D, S, mid, right, (z + m) % half);
#pragma omp taskwait

        bool s = ((z % half + m) >= half) ^ (z >= half);
        int threshold = (z + m) % half;
        if (n >= 8192)
        {
#pragma omp taskloop grainsize(1024) shared(D) firstprivate(left, half, s, threshold)
            for (int i = 0; i < half; ++i)
            {
                bool b = s ^ (i >= threshold);
                packedSwapRows(D, left + i, left + i + half, b);
            }
        }
        else
        {
            for (int i = 0; i < half; ++i)
            {
                bool b = s ^ (i >= threshold);
                packedSwapRows(D, left + i, left + i + half, b);
            }
        }
    }
}

static void packedORCPARTask(PackedFlatTable &D, const vector<int> &S, int left, int right)
{
    int n = right - left;
    if (n == 0)
        return;

    int n1 = 1;
    while ((n1 << 1) <= n)
        n1 <<= 1;
    int n2 = n - n1;
    int m = S[left + n2] - S[left];

#pragma omp task shared(D, S) firstprivate(left, n2) if (n >= 8192 && n2 > 0)
    packedORCPARTask(D, S, left, left + n2);
#pragma omp task shared(D, S) firstprivate(left, right, n1, n2, m) if (n >= 8192)
    packedOROCPARTask(D, S, left + n2, right, (n1 - n2 + m) % n1);
#pragma omp taskwait

    if (n2 >= 8192)
    {
#pragma omp taskloop grainsize(1024) shared(D) firstprivate(left, n1, n2, m)
        for (int i = 0; i < n2; ++i)
            packedSwapRows(D, left + i, left + n1 + i, i >= m);
    }
    else
    {
        for (int i = 0; i < n2; ++i)
            packedSwapRows(D, left + i, left + n1 + i, i >= m);
    }
}

static void packedORCompactPar(PackedFlatTable &D, const vector<int> &M, int threads)
{
    PrimitiveProfileScope primitiveScope(PrimitiveCompact);
    if (D.rows == 0)
        return;
    if (threads <= 1 || D.rows < 8192)
    {
        packedORCompact(D, M, 0, D.rows);
        return;
    }

    vector<int> S = prefixSumMarksPar(M, threads);
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
        packedORCPARTask(D, S, 0, D.rows);
    }
}

// Fixed lexicographic bitonic network. Padding has an independent flag, so
// every signed integer key (including INT_MIN/INT_MAX) is legitimate data.
// Neither key magnitudes nor sorted row indices choose memory addresses.
template <class Storage>
static Storage fixedNetworkSort(Storage table, const vector<int> &keyColumns,
                                bool direction, int sortThreads)
{
    PrimitiveProfileScope primitiveScope(PrimitiveSort);
    const int n = table.rows;
    if (n == 0)
        return table;
    const int threads = sortThreads > 0 ? sortThreads : omp_get_max_threads();
    int length = 1;
    while (length < n)
        length <<= 1;
    table.resizeRows(length);
    vector<int> padding(length, 1);
    for (int i = 0; i < n; ++i)
        padding[i] = 0;
    for (int block = 2; block <= length; block <<= 1)
    {
        for (int stride = block >> 1; stride > 0; stride >>= 1)
        {
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int pair = 0; pair < length / 2; ++pair)
            {
                const int left = (pair / stride) * (stride * 2) + pair % stride;
                const int right = left + stride;
                bool less = padding[left] < padding[right];
                bool greater = padding[left] > padding[right];
                bool equal = padding[left] == padding[right];
                for (int col : keyColumns)
                {
                    const int a = table.get(left, col), b = table.get(right, col);
                    less = less | (equal & (direction ? a < b : a > b));
                    greater = greater | (equal & (direction ? a > b : a < b));
                    equal = equal & (a == b);
                }
                const bool exchange = ((left & block) == 0) ? greater : less;
                const uint32_t mask = 0u - static_cast<uint32_t>(exchange);
                for (int col = 0; col < table.cols; ++col)
                {
                    const uint32_t a = static_cast<uint32_t>(table.get(left, col));
                    const uint32_t b = static_cast<uint32_t>(table.get(right, col));
                    const uint32_t diff = (a ^ b) & mask;
                    table.set(left, col, static_cast<int>(a ^ diff));
                    table.set(right, col, static_cast<int>(b ^ diff));
                }
                const int diff = (padding[left] ^ padding[right]) & static_cast<int>(mask);
                padding[left] ^= diff;
                padding[right] ^= diff;
            }
        }
        if (block == length)
            break;
    }
    table.resizeRows(n);
    return table;
}

static FlatTable sortFlatTableIdx(FlatTable table, const vector<int> &keys,
                                 bool direction, int threads = 0)
{
    return fixedNetworkSort(std::move(table), keys, direction, threads);
}

static PackedFlatTable sortPackedTableIdx(PackedFlatTable table, const vector<int> &keys,
                                         bool direction, int threads = 0)
{
    return fixedNetworkSort(std::move(table), keys, direction, threads);
}
static Table sortTableIdx(Table table, vector<int> keyColumns, bool direction, int sortThreads = 0)
{
    if (table.empty())
        return table;
    int rows = (int)table.size();
    int cols = (int)table[0].size();
    FlatTable flat(rows, cols);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
            flat.at(i, j) = table[i][j];
    }

    flat = sortFlatTableIdx(std::move(flat), keyColumns, direction, sortThreads);

    Table sorted(rows, vector<int>(cols));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
            sorted[i][j] = flat.at(i, j);
    }
    return sorted;
}

static Table flatToTable(const FlatTable &flat)
{
    Table table(flat.rows, vector<int>(flat.cols));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < flat.rows; ++i)
    {
        for (int k = 0; k < flat.cols; ++k)
            table[i][k] = flat.at(i, k);
    }
    return table;
}

static FlatTable sortTempResultByPFlat(FlatTable flat, int keyCols, int sortThreads = 0)
{
    PrimitiveProfileScope primitiveScope(PrimitiveSort);
    int n = flat.rows;
    if (n == 0)
        return flat;

    int cols = keyCols + 1;
    int len = 1;
    while (len < n)
        len <<= 1;
    int threads = sortThreads > 0 ? sortThreads : omp_get_max_threads();

    flat.data.resize((size_t)len * cols, 0);
    flat.rows = len;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = n; i < len; ++i)
        flat.at(i, keyCols) = INT_MAX;

#pragma omp parallel num_threads(threads)
    {
        for (int k = 2; k <= len; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                int pairCount = len >> 1;
                int blockSize = j << 1;
#pragma omp for schedule(static)
                for (int pair = 0; pair < pairCount; ++pair)
                {
                    int block = (pair / j) * blockSize;
                    int i = block + (pair % j);
                    int l = i + j;
                    bool asc = ((i & k) == 0);
                    flatSwapRows(flat, i, l, (flat.at(i, keyCols) > flat.at(l, keyCols)) == asc);
                }
            }
        }
    }

    flat.rows = n;
    flat.data.resize((size_t)n * cols);
    return flat;
}

Table JFYanDown(vector<Table> &tables, const vector<int> &parent, int root,
              const vector<int> &joinColInParent, const vector<int> &joinColInChild, const vector<int> &tableKeys)
{
    g_lastJFYanDownParallelMs = 0.0;
    g_lastJFYanOutputValidity.clear();
    if (tables[root].empty()) // public output length zero
        return Table();
    memoryProfileBeginPositionPropagation();
    int n = tables.size();
    vector<FlatTable> tempResult;
    tempResult.resize(tables.size());

    /*
    Step 2: Top-down expansion
    */

    // Build children list from parent array
    vector<vector<int>> children(n);
    for (int i = 0; i < n; i++)
        if (parent[i] != -1)
            children[parent[i]].push_back(i);

    // BFS from root to get top-down order
    vector<int> order;
    queue<int> q;
    q.push(root);
    while (!q.empty())
    {
        int cur = q.front();
        q.pop();
        order.push_back(cur);
        for (int c : children[cur])
            q.push(c);
    }

    vector<double> nodeWorkMs(n, 0.0);
    vector<double> edgeWorkMs(n, 0.0);

    // For each internal node, call ParallelOExpand(parent, child) top-down
    for (int oi = 0; oi < (int)order.size(); oi++)
    {
        int p = order[oi];
        if (children[p].empty())
            continue;
        double sgxNodeStart = sgxProfileNowMs();

        // printf("================[JFYanDown] parent node=%d  rows=%d================\n", p, (int)tables[p].size());

        // // print parent table
        // cout << "Parent table:\n";
        // for (const auto &row : tables[p])
        // {
        //     cout << "  [";
        //     for (int i = 0; i < (int)row.size(); i++)
        //     {
        //         if (i)
        //             cout << ", ";
        //         cout << row[i];
        //     }
        //     cout << "]\n";
        // }

        /*
        Step 4: Compute local index for each copy within its group
        */
        // p(key1, key2, ..., u1,u2,...,delta, b, P, lambda)
#ifndef SGX_ENCLAVE_BUILD
        auto _t0 = high_resolution_clock::now();
#endif
        // Algorithm 5 line 2: sort by (state, tid). In this layout state=(b,lambda)
        // and tid is represented by the final output position P.
        if (p != root)
        {
            vector<int> stateTidCols = {(int)tables[p][0].size() - 3,
                                        (int)tables[p][0].size() - 1,
                                        (int)tables[p][0].size() - 2};
            tables[p] = sortTableIdx(std::move(tables[p]), stateTidCols, true);
        }
#ifndef SGX_ENCLAVE_BUILD
        auto _t1 = high_resolution_clock::now();
        printf("  [node %d] sort-by-state:  %.2f ms\n", p, duration<double, milli>(_t1 - _t0).count());
#endif
        double sgxAfterSortState = sgxProfileNowMs();

        const int b_pos = (int)tables[p][0].size() - 3;
        const int lambda_pos = (int)tables[p][0].size() - 1;
        const int P_pos = (int)tables[p][0].size() - 2;
#ifndef SGX_ENCLAVE_BUILD
        auto _t2 = high_resolution_clock::now();
#endif
        vector<int> sameState(tables[p].size(), 0);
        vector<int> rawStep(tables[p].size(), 0);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)tables[p].size(); ++i)
        {
            if (i > 0)
            {
                sameState[i] = (tables[p][i][b_pos] == tables[p][i - 1][b_pos]) &
                               (tables[p][i][lambda_pos] == tables[p][i - 1][lambda_pos]);
            }
            const int previous = i > 0 ? i - 1 : i;
            rawStep[i] = selectValue(sameState[i],
                tables[p][i][P_pos] - tables[p][previous][P_pos],
                tables[p][i][P_pos] - tables[p][i][b_pos]);
        }

        vector<int> idxRaw = Aggtree(rawStep, sameState, 0).PrefixTreeRun();
        vector<int> idx(tables[p].size(), 0);
#pragma omp parallel for
        for (int i = 0; i < (int)tables[p].size(); ++i)
        {
            int lambda = tables[p][i][lambda_pos];
            const bool valid = tables[p][i][tables[p][i].size() - 4] > 0;
            const int divisor = selectValue(lambda > 0, lambda, 1);
            idx[i] = selectValue(valid, idxRaw[i] / divisor, 0);
        }
        vector<int>().swap(rawStep);
        vector<int>().swap(sameState);
        vector<int>().swap(idxRaw);
#ifndef SGX_ENCLAVE_BUILD
        auto _t3 = high_resolution_clock::now();
        printf("  [node %d] aggtree-idx:    %.2f ms\n", p, duration<double, milli>(_t3 - _t2).count());
#endif
        double sgxAfterIdx = sgxProfileNowMs();

        int m = (int)children[p].size();
        const int parentRows = (int)tables[p].size();
        const int origCols = (int)tables[p][0].size() - m - 4;

        const size_t siblingCells = static_cast<size_t>(parentRows) * static_cast<size_t>(m);
        vector<int> scFlat(siblingCells, 1);
        // Children of one parent row form one fixed-size public segment. A
        // direct reverse scan is exactly the exclusive suffix product that
        // the generic tree computed, without building five extra level trees
        // over parentRows*m entries.
        {
            PrimitiveProfileScope suffixScope(PrimitiveAggtree);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < parentRows; ++i)
            {
                const size_t base = static_cast<size_t>(i) * static_cast<size_t>(m);
                long long suffix = 1;
                for (int j = m - 1; j >= 0; --j)
                {
                    const size_t pos = base + static_cast<size_t>(j);
                    scFlat[pos] = static_cast<int>(suffix);
                    const int u = tables[p][i][origCols + j];
                    const int nonnegativeU = selectValue(u > 0, u, 0);
                    const long long product = suffix * nonnegativeU;
                    suffix = selectValue(product > INT_MAX, INT_MAX, static_cast<int>(product));
                }
            }
        }

        vector<int> pcFlat(siblingCells, 0);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < parentRows; ++i)
        {
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(m);
            for (int j = 0; j < m; ++j)
            {
                const size_t pos = base + static_cast<size_t>(j);
                int sc_j = scFlat[pos];
                int u_j = tables[p][i][origCols + j];
                const int safeSc = selectValue(sc_j > 0, sc_j, 1);
                const int safeU = selectValue(u_j > 0, u_j, 1);
                pcFlat[pos] = (idx[i] / safeSc) % safeU;
            }
        }

#ifndef SGX_ENCLAVE_BUILD
        auto _t3_5 = high_resolution_clock::now();
        printf("  [node %d] suffix-sc/pc:  %.2f ms\n", p, duration<double, milli>(_t3_5 - _t3).count());
        nodeWorkMs[p] = duration<double, milli>(_t3_5 - _t0).count();
#endif
        double sgxAfterSuffix = sgxProfileNowMs();
#ifdef SGX_ENCLAVE_BUILD
        nodeWorkMs[p] = sgxAfterSuffix - sgxNodeStart;
#endif
        // Each child subProcess is an independent branch of the critical path.
        // Measure each branch with the full inner OpenMP team and let the
        // critical-path recurrence below take max(child). Running all children
        // at once would make them share the finite local thread pool and would
        // overestimate the per-branch critical time.
        int childCount = (int)children[p].size();
        int edgeMaxThreads = omp_get_max_threads();
#ifdef SGX_ENCLAVE_BUILD
        int edgeWorkers = chooseJFYanDownWorkerCount(tables, children[p], parentRows, edgeMaxThreads);
        int edgeInnerThreads = splitInnerThreadCount(edgeMaxThreads, edgeWorkers);
#else
        int edgeMaxWorkers = std::max(1, edgeMaxThreads / 2);
        int edgeWorkers = std::max(1, std::min(childCount, edgeMaxWorkers));
        int edgeInnerThreads = splitInnerThreadCount(edgeMaxThreads, edgeWorkers);
#endif
        if (sgxProfileEnabled())
        {
            char buf[320];
            snprintf(buf, sizeof(buf),
                     "  [JFYanDown node p=%d] sortState=%.2f ms idx=%.2f ms suffixPc=%.2f ms rows=%d children=%d edgeWorkers=%d innerThreads=%d",
                     p,
                     sgxAfterSortState - sgxNodeStart,
                     sgxAfterIdx - sgxAfterSortState,
                     sgxAfterSuffix - sgxAfterIdx,
                     parentRows,
                     m,
                     edgeWorkers,
                     edgeInnerThreads);
            sgxProfilePrint(buf);
        }
        int oldEdgeActiveLevels = omp_get_max_active_levels();
        if (edgeWorkers > 1)
            omp_set_max_active_levels(std::max(oldEdgeActiveLevels, 2));

        // At p=16 the SGX schedule assigns the full inner team to one sibling
        // branch at a time.  Profiling the branches separately lets us report
        // both the measured peak and a conservative all-siblings-concurrent
        // memory bound without changing that schedule.
        memoryProfileBeginBranchGroup(edgeWorkers == 1);

// p(key1, key2, ..., u1,u2,...,delta, b, P, lambda, idx, sc1,pc1,...)
#pragma omp parallel for num_threads(edgeWorkers)
        for (int j = 0; j < childCount; j++)
        {
            memoryProfileBeginBranch();
            int c = children[p][j];
            double sgxEdgeStart = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
            auto _ts0 = high_resolution_clock::now();
#endif
            // get child rank without appending a column to every child row.
            vector<int> childRank = getRankValues(tables[c], joinColInChild[c], children[c].empty(), edgeInnerThreads);
            double sgxAfterRank = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
            auto _ts0_5 = high_resolution_clock::now();
#endif
            tempResult[c] = subProcessFlat(tables[p], joinColInParent[c], tables[c], joinColInChild[c],
                                           childRank, idx, scFlat, pcFlat, m, c, children[c].empty(), j, tableKeys[c],
                                           edgeInnerThreads, static_cast<int>(children[c].size()));
            memoryProfileEndBranch();
            double sgxAfterSub = sgxProfileNowMs();
#ifdef SGX_ENCLAVE_BUILD
            edgeWorkMs[c] = sgxAfterSub - sgxEdgeStart;
            if (sgxProfileEnabled())
            {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "  [JFYanDown edge p=%d c=%d] getRank=%.2f ms subProcess=%.2f ms total=%.2f ms",
                         p, c,
                         sgxAfterRank - sgxEdgeStart,
                         sgxAfterSub - sgxAfterRank,
                         sgxAfterSub - sgxEdgeStart);
#pragma omp critical(JFYanDownEdgeProfile)
                sgxProfilePrint(buf);
            }
#endif
#ifndef SGX_ENCLAVE_BUILD
            auto _ts1 = high_resolution_clock::now();
            edgeWorkMs[c] = duration<double, milli>(_ts1 - _ts0).count();
            printf("  [node %d -> child %d] getRank: %.2f ms  subProcess: %.2f ms\n",
                   p, c, duration<double, milli>(_ts0_5 - _ts0).count(),
                   duration<double, milli>(_ts1 - _ts0_5).count());
#endif

            // print tempResult[c]
            // cout << "tempResult[" << c << "] after sorting by child's join key:\n";
            // for (const auto &row : tempResult[c])
            // {
            //     cout << "  [";
            //     for (int i = 0; i < (int)row.size(); i++)
            //     {
            //         if (i)
            //             cout << ", ";
            //         cout << row[i];
            //     }
            //     cout << "]\n";
            // }

            // // print child
            // cout << "Child " << c << " after subProcess:\n";
            // for (const auto &row : tables[c])
            // {
            //     cout << "  [";
            //     for (int i = 0; i < (int)row.size(); i++)
            //     {
            //         if (i)
            //             cout << ", ";
            //         cout << row[i];
            //     }
            //     cout << "]\n";
            // }
            // break;
            // return Table();
        }
        memoryProfileEndBranchGroup();
        if (edgeWorkers > 1)
            omp_set_max_active_levels(oldEdgeActiveLevels);
        // break;
    }

    vector<double> criticalMs(n, 0.0);
    // Submitted timing model: node-local setup is serial, while sibling edges
    // are independent.  Pair each edge with its dependent child subtree before
    // taking the sibling maximum so dependencies on the same branch are added.
    for (int oi = (int)order.size() - 1; oi >= 0; --oi)
    {
        int u = order[oi];
        double bestChild = 0.0;
        for (int c : children[u])
            bestChild = max(bestChild, edgeWorkMs[c] + criticalMs[c]);
        criticalMs[u] = nodeWorkMs[u] + bestChild;
    }
    double assignmentCriticalMs = criticalMs[root];

    memoryProfileSetCopyTableBytes(flatTableStorageBytes(tempResult));
    memoryProfileEndPositionPropagation();

    // root was never processed by subProcess (it has a different column layout),
    // so sort only non-root nodes whose tempResult = [keys, P]
#ifndef SGX_ENCLAVE_BUILD
    auto _final_sort_t0 = high_resolution_clock::now();
#endif
    vector<double> finalSortMs(tempResult.size(), 0.0);
    double sgxFinalSortStart = sgxProfileNowMs();
    int finalSortCount = 0;
    for (int c = 0; c < (int)tempResult.size(); ++c)
    {
        if (c != root && tempResult[c].rows > 0)
            ++finalSortCount;
    }
    int maxThreads = omp_get_max_threads();
    int finalSortWorkers = chooseFinalSortWorkerCount(tempResult, root, finalSortCount, maxThreads);
    int finalSortInnerThreads = splitInnerThreadCount(maxThreads, finalSortWorkers);
    int oldFinalSortActiveLevels = omp_get_max_active_levels();
    if (finalSortWorkers > 1)
        omp_set_max_active_levels(std::max(oldFinalSortActiveLevels, 2));
#pragma omp parallel for num_threads(finalSortWorkers) schedule(static)
    for (int c = 0; c < (int)tempResult.size(); c++)
    {
        if (c == root)
            continue;
        if (tempResult[c].rows <= 0)
            continue;
        double sgxNodeSortStart = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
        auto _node_sort_t0 = high_resolution_clock::now();
#endif
        tempResult[c] = sortTempResultByPFlat(std::move(tempResult[c]), tableKeys[c], finalSortInnerThreads);
        double sgxNodeSortEnd = sgxProfileNowMs();
#ifdef SGX_ENCLAVE_BUILD
        finalSortMs[c] = sgxNodeSortEnd - sgxNodeSortStart;
        if (sgxProfileEnabled())
        {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "  [JFYanDown finalSort c=%d] rows=%d ms=%.2f",
                     c, tempResult[c].rows, sgxNodeSortEnd - sgxNodeSortStart);
#pragma omp critical(JFYanDownFinalSortProfile)
            sgxProfilePrint(buf);
        }
#endif
#ifndef SGX_ENCLAVE_BUILD
        auto _node_sort_t1 = high_resolution_clock::now();
        finalSortMs[c] = duration<double, milli>(_node_sort_t1 - _node_sort_t0).count();
#endif
    }
    if (finalSortWorkers > 1)
        omp_set_max_active_levels(oldFinalSortActiveLevels);
    double sgxFinalSortEnd = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
    auto _final_sort_t1 = high_resolution_clock::now();
#endif

    double finalSortCriticalMs = 0.0;
    for (double v : finalSortMs)
        finalSortCriticalMs = max(finalSortCriticalMs, v);

    // Every non-root copy table must be ready before the fixed-access output
    // assembly, so independent sorts take max and materialization follows it.
#ifndef SGX_ENCLAVE_BUILD
    auto _materialize_t0 = high_resolution_clock::now();
#endif
    double sgxMaterializeStart = sgxProfileNowMs();
    memoryProfileSetCopyTableBytes(flatTableStorageBytes(tempResult));
    int numRows = (int)tables[root].size();
    int totalKeys = 0;
    for (int i = 0; i < (int)tempResult.size(); i++)
        totalKeys += tableKeys[i];
    Table result(numRows, vector<int>(totalKeys));
    g_lastJFYanOutputValidity.assign(numRows, 0);
#pragma omp parallel for
    for (int j = 0; j < numRows; j++)
    {
        g_lastJFYanOutputValidity[j] = tables[root][j][tables[root][j].size() - 4] > 0;
        int offset = 0;
        for (int i = 0; i < (int)tempResult.size(); i++)
        {
            if (i == root)
            {
                for (int k = 0; k < tableKeys[i]; k++)
                    result[j][offset + k] = tables[root][j][k];
            }
            else
            {
                for (int k = 0; k < tableKeys[i]; k++)
                    result[j][offset + k] = tempResult[i].at(j, k);
            }
            offset += tableKeys[i];
        }
    }
    memoryProfileSetFinalOutputBytes(memoryProfileTableBytes(result));
    double sgxMaterializeEnd = sgxProfileNowMs();
#ifndef SGX_ENCLAVE_BUILD
    auto _materialize_t1 = high_resolution_clock::now();
    double finalMaterializeMs = duration<double, milli>(_materialize_t1 - _materialize_t0).count();
    double finalSortWallMs = duration<double, milli>(_final_sort_t1 - _final_sort_t0).count();
#else
    double finalMaterializeMs = sgxMaterializeEnd - sgxMaterializeStart;
#endif
    double finalStageCriticalMs = finalSortCriticalMs + finalMaterializeMs;
    g_lastJFYanDownParallelMs = assignmentCriticalMs + finalStageCriticalMs;
#ifndef SGX_ENCLAVE_BUILD
    printf("  [JFYanDown parallel-estimate] assignmentCritical=%.2f ms  finalSortCritical=%.2f ms  finalSortWall=%.2f ms  finalMaterialize=%.2f ms  total=%.2f ms\n",
           assignmentCriticalMs, finalSortCriticalMs, finalSortWallMs, finalMaterializeMs, g_lastJFYanDownParallelMs);
#endif
    if (sgxProfileEnabled())
    {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "  [JFYanDown final] sortTotal=%.2f ms materialize=%.2f ms rows=%d cols=%d sortWorkers=%d innerThreads=%d",
                 sgxFinalSortEnd - sgxFinalSortStart,
                 sgxMaterializeEnd - sgxMaterializeStart,
                 numRows,
                 totalKeys,
                 finalSortWorkers,
                 finalSortInnerThreads);
        sgxProfilePrint(buf);
    }
    return result;
    // return Table();
}

//  childNum is the index of this child in its parent's children list.
//  keyCols is used to restore child table keys
static FlatTable subProcessFlat(const Table &parent, int joinColInParent, Table &child, int joinColInChild,
                                const vector<int> &childRank, const vector<int> &parentIdx,
                                const vector<int> &scFlat, const vector<int> &pcFlat,
                                int siblingCount, int c, bool isLeafNode, int childNum, int keyCols,
                                int workerThreads, int childArity)
{
    if (parent.empty())
    {
#ifndef SGX_ENCLAVE_BUILD
        cerr << "Error: parent table is empty in subProcess. This should not happen if bottom-up semi-join is correct." << endl;
#endif
        return FlatTable();
    }

#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t0 = high_resolution_clock::now();
#endif
    double sgxSpT0 = sgxProfileNowMs();

    const int childIdx = childNum; // index of Rc among Rp's children
    const int numU = isLeafNode ? 0 : (childArity >= 0 ? childArity :
        (child.empty() ? 0 : max(0, (int)child[0].size() - keyCols - 1)));
    const int effNumU = numU;

    // M(key, ord, tid, child-keys..., P, lambda, idx, s_c, p_c, delta_c, u_c..., rank_c)
    // tid: 0 for Rc header tuples and 1 for Rp copies, so Rc precedes Rp on (key,ord) ties.
    const int M_KEY = 0;
    const int M_ORD = 1;
    const int M_TID = 2;
    const int M_KEYS = 3;
    const int M_P = M_KEYS + keyCols;
    const int M_LAMBDA = M_P + 1;
    const int M_IDX = M_P + 2;
    const int M_SC = M_P + 3;
    const int M_PC = M_P + 4;
    const int M_DELTA = M_P + 5;
    const int M_U = M_P + 6;
    const int M_RANK = M_U + effNumU;
    const int M_DUMMY = M_RANK + 1;
    const int M_BASE = M_DUMMY + 1;
    const int rowSize = M_BASE + 1;

    const int mRows = (int)(parent.size() + child.size()) + 1;
    PackedFlatTable M(mRows, rowSize);
    const int parentRowLen = (int)parent[0].size();
    const int origCols = parentRowLen - siblingCount - 4;
    const int delta_col = origCols + siblingCount;
    const int P_col = delta_col + 2;
    const int lambda_col = delta_col + 3;

#pragma omp parallel for
    for (int i = 0; i < (int)parent.size(); i++)
    {
        int sc_j = scFlat[(size_t)i * siblingCount + childIdx];
        int pc_j = pcFlat[(size_t)i * siblingCount + childIdx];

        const bool dummy = parent[i][delta_col] == 0;
        M.set(i, M_DUMMY, dummy);
        M.set(i, M_KEY, selectValue(dummy, 0, parent[i][joinColInParent]));
        M.set(i, M_ORD, selectValue(dummy, 0, pc_j));
        M.set(i, M_TID, 1);
        M.set(i, M_P, parent[i][P_col]);
        M.set(i, M_LAMBDA, selectValue(dummy, 1, parent[i][lambda_col]));
        M.set(i, M_BASE, selectValue(dummy, parent[i][P_col], parent[i][delta_col + 1]));
        M.set(i, M_IDX, selectValue(dummy, 0, parentIdx[i]));
        M.set(i, M_SC, selectValue(dummy, 1, sc_j));
        M.set(i, M_PC, selectValue(dummy, 0, pc_j));
    }
#pragma omp parallel for
    for (int i = 0; i < (int)child.size(); i++)
    {
        int row = (int)parent.size() + i;
        M.set(row, M_KEY, child[i][joinColInChild]);
        M.set(row, M_ORD, childRank[i]);
        M.set(row, M_TID, 0);
        for (int k = 0; k < keyCols; ++k)
            M.set(row, M_KEYS + k, child[i][k]);
        M.set(row, M_DELTA, isLeafNode ? 1 : child[i].back());
        for (int k = 0; k < effNumU; k++)
            M.set(row, M_U + k, child[i][keyCols + k]);
        M.set(row, M_RANK, childRank[i]);
    }
    // A separate namespace bit prevents dummy copies from matching any real
    // key, even if a real key equals the dummy payload's zero value.
    const int dummyHeader = mRows - 1;
    M.set(dummyHeader, M_DUMMY, 1);
    M.set(dummyHeader, M_TID, 0);
    for (int k = 0; k < effNumU; ++k)
        M.set(dummyHeader, M_U + k, 1);

#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t1 = high_resolution_clock::now();
    printf("    [subProcess c=%d] build-M:      %.2f ms  (M.size=%d)\n", c, duration<double, milli>(_sp_t1 - _sp_t0).count(), M.rows);
#endif
    double sgxSpT1 = sgxProfileNowMs();

    int threads = workerThreads > 0 ? workerThreads : omp_get_max_threads();
    M = sortPackedTableIdx(std::move(M), {M_DUMMY, M_KEY, M_ORD, M_TID}, true, threads);
#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t2 = high_resolution_clock::now();
    printf("    [subProcess c=%d] sort-align:   %.2f ms\n", c, duration<double, milli>(_sp_t2 - _sp_t1).count());
#endif
    double sgxSpT2 = sgxProfileNowMs();

    vector<int> Bm(M.rows, 0);
#pragma omp parallel for schedule(static)
    for (int i = 1; i < M.rows; i++)
        Bm[i] = (M.get(i, M_TID) == 1) &
                (M.get(i, M_KEY) == M.get(i - 1, M_KEY)) &
                (M.get(i, M_DUMMY) == M.get(i - 1, M_DUMMY));

#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t3 = high_resolution_clock::now();
    printf("    [subProcess c=%d] mark-match:   %.2f ms\n", c, duration<double, milli>(_sp_t3 - _sp_t2).count());
#endif
    double sgxSpT3 = sgxProfileNowMs();

    const int slimDim = keyCols + effNumU + 2; // keys, u-values, delta, rank
    vector<int> M_slim((size_t)M.rows * slimDim);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M.rows; i++)
    {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(slimDim);
        for (int k = 0; k < keyCols; k++)
            M_slim[base + k] = M.get(i, M_KEYS + k);
        for (int k = 0; k < effNumU; k++)
            M_slim[base + keyCols + k] = M.get(i, M_U + k);
        M_slim[base + keyCols + effNumU] = M.get(i, M_DELTA);
        M_slim[base + keyCols + effNumU + 1] = M.get(i, M_RANK);
    }
#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t4 = high_resolution_clock::now();
    printf("    [subProcess c=%d] slim-build:   %.2f ms\n", c, duration<double, milli>(_sp_t4 - _sp_t3).count());
#endif
    double sgxSpT4 = sgxProfileNowMs();

    DupAggtree dupAgg(M_slim, M.rows, slimDim, Bm);
    M_slim = dupAgg.DupAggtreeRunFlat();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < M.rows; i++)
    {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(slimDim);
        for (int k = 0; k < keyCols; k++)
            M.set(i, M_KEYS + k, M_slim[base + k]);
        for (int k = 0; k < effNumU; k++)
            M.set(i, M_U + k, M_slim[base + keyCols + k]);
        M.set(i, M_DELTA, M_slim[base + keyCols + effNumU]);
        M.set(i, M_RANK, M_slim[base + keyCols + effNumU + 1]);
    }
#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t5 = high_resolution_clock::now();
    printf("    [subProcess c=%d] DupAggtree:   %.2f ms\n", c, duration<double, milli>(_sp_t5 - _sp_t4).count());
#endif
    double sgxSpT5 = sgxProfileNowMs();

    // Retain every parent position (real or dummy), not just matched rows.
    vector<int> keep(M.rows, 0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M.rows; ++i)
        keep[i] = M.get(i, M_TID) == 1;
    packedORCompactPar(M, keep, threads);
    M.rows = (int)parent.size();
#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t6 = high_resolution_clock::now();
    printf("    [subProcess c=%d] compact-M:    %.2f ms\n", c, duration<double, milli>(_sp_t6 - _sp_t5).count());
#endif
    double sgxSpT6 = sgxProfileNowMs();

    const int resultRowLen = keyCols + 1;
    const int cRowLen = keyCols + numU + 4;
    FlatTable result(M.rows, resultRowLen);
    Table new_child;
    if (!isLeafNode)
        new_child.assign(M.rows, vector<int>(cRowLen));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M.rows; i++)
    {
        const int new_p = M.get(i, M_P);

        for (int k = 0; k < keyCols; k++)
            result.at(i, k) = M.get(i, M_KEYS + k);
        result.at(i, keyCols) = new_p;

        if (!isLeafNode)
        {
            const int sc_i = M.get(i, M_SC);
            const int alpha = M.get(i, M_IDX) % selectValue(sc_i > 0, sc_i, 1);
            const int beta = (M.get(i, M_PC) - M.get(i, M_RANK)) * sc_i;
            const int inheritedLambda = M.get(i, M_LAMBDA);
            const int residual = (new_p - M.get(i, M_BASE)) %
                selectValue(inheritedLambda > 0, inheritedLambda, 1);
            const int new_b = new_p - residual - inheritedLambda * (alpha + beta);
            const int new_lambda = M.get(i, M_LAMBDA) * sc_i;
            const int new_delta = M.get(i, M_DELTA);

            for (int k = 0; k < keyCols; k++)
                new_child[i][k] = M.get(i, M_KEYS + k);
            for (int k = 0; k < numU; k++)
                new_child[i][keyCols + k] = M.get(i, M_U + k);
            new_child[i][keyCols + numU] = new_delta;
            new_child[i][keyCols + numU + 1] = new_b;
            new_child[i][keyCols + numU + 2] = new_p;
            new_child[i][keyCols + numU + 3] = new_lambda;
        }
    }

#ifndef SGX_ENCLAVE_BUILD
    auto _sp_t7 = high_resolution_clock::now();
    printf("    [subProcess c=%d] build-result: %.2f ms\n", c, duration<double, milli>(_sp_t7 - _sp_t6).count());
#endif
    double sgxSpT7 = sgxProfileNowMs();

    if (sgxProfileEnabled())
    {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "    [JFYanDown subProcess c=%d] buildM=%.2f ms sort=%.2f ms mark=%.2f ms slim=%.2f ms dup=%.2f ms compact=%.2f ms buildResult=%.2f ms total=%.2f ms rows=%d",
                 c,
                 sgxSpT1 - sgxSpT0,
                 sgxSpT2 - sgxSpT1,
                 sgxSpT3 - sgxSpT2,
                 sgxSpT4 - sgxSpT3,
                 sgxSpT5 - sgxSpT4,
                 sgxSpT6 - sgxSpT5,
                 sgxSpT7 - sgxSpT6,
                 sgxSpT7 - sgxSpT0,
                 M.rows);
        sgxProfilePrint(buf);
    }

    if (!isLeafNode)
        child = std::move(new_child); // write updated child back through reference
    return result;
}

Table subProcess(const Table &parent, int joinColInParent, Table &child, int joinColInChild,
                 const vector<int> &childRank, const vector<int> &parentIdx,
                 const vector<int> &scFlat, const vector<int> &pcFlat,
                 int siblingCount, int c, bool isLeafNode, int childNum, int keyCols)
{
    return flatToTable(subProcessFlat(parent, joinColInParent, child, joinColInChild,
                                      childRank, parentIdx, scFlat, pcFlat,
                                      siblingCount, c, isLeafNode, childNum, keyCols));
}

vector<int> getRankValues(Table &table, int joinCols, bool isLeafNode, int sortThreads)
{
    if (table.empty())
        return vector<int>();
    double sgxRankStart = sgxProfileNowMs();
    // table(key1, key2, ..., keyN,u1,u2,...,delta)
    vector<int> B(table.size(), 0);
    vector<int> delta_c(table.size(), 0);
    // child(key1, key2, ..., keyN, u1, u2, ..., delta)
    table = sortTable(table, {joinCols, (int)table[0].size() - 1}, false, sortThreads);
    double sgxAfterSort = sgxProfileNowMs();
    B[0] = 0, delta_c[0] = (isLeafNode ? 1 : table[0].back());
#pragma omp parallel for
    for (int i = 1; i < table.size(); i++)
    {
        B[i] = (table[i][joinCols] == table[i - 1][joinCols]);
        delta_c[i] = (isLeafNode ? 1 : table[i].back());
    }

    vector<int> rank = Aggtree(delta_c, B, 1).PrefixTreeRun();
    double sgxAfterAgg = sgxProfileNowMs();
    if (sgxProfileEnabled())
    {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "    [JFYanDown getRank] rows=%d sort=%.2f ms agg=%.2f ms append=0.00 ms total=%.2f ms",
                 (int)table.size(),
                 sgxAfterSort - sgxRankStart,
                 sgxAfterAgg - sgxAfterSort,
                 sgxAfterAgg - sgxRankStart);
        sgxProfilePrint(buf);
    }
    return rank;
}

void getRank(Table &table, int joinCols, bool isLeafNode)
{
    vector<int> rank = getRankValues(table, joinCols, isLeafNode);
    const int oldCols = (int)table[0].size();
#pragma omp parallel for
    for (int i = 0; i < (int)table.size(); i++)
    {
        table[i].resize(oldCols + 1);
        table[i][oldCols] = rank[i];
    }
}

Table sortTable(Table table, vector<int> keyColumns, bool direction, int sortThreads)
{
    return sortTableIdx(std::move(table), std::move(keyColumns), direction, sortThreads);
}
 
