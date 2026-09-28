#include "../include/NonObliviousJFYan.h"
#include "../include/sgx_profile.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <queue>
#include <unordered_map>
#include <utility>

namespace
{
using Count = long long;
using Table = NonObliviousTable;

struct MatchBucket
{
    std::vector<int> tupleIds;
    std::vector<Count> cumulativeEnds;
    Count total = 0;
};

using EdgeIndex = std::unordered_map<int, MatchBucket>;

bool checkedAdd(Count a, Count b, Count &out)
{
    if (a < 0 || b < 0 || a > LLONG_MAX - b)
        return false;
    out = a + b;
    return true;
}

bool checkedMultiply(Count a, Count b, Count &out)
{
    if (a < 0 || b < 0 || (a != 0 && b > LLONG_MAX / a))
        return false;
    out = a * b;
    return true;
}

bool validateInput(const std::vector<Table> &tables,
                   const std::vector<int> &parent,
                   int root,
                   const std::vector<int> &joinColInParent,
                   const std::vector<int> &joinColInChild,
                   const std::vector<int> &tableKeys,
                   std::vector<std::vector<int>> &children,
                   std::vector<int> &topDownOrder)
{
    const int n = static_cast<int>(tables.size());
    if (n <= 0 || root < 0 || root >= n ||
        static_cast<int>(parent.size()) != n ||
        static_cast<int>(joinColInParent.size()) != n ||
        static_cast<int>(joinColInChild.size()) != n ||
        static_cast<int>(tableKeys.size()) != n ||
        parent[root] != -1)
    {
        return false;
    }

    children.assign(n, {});
    for (int u = 0; u < n; ++u)
    {
        if (tableKeys[u] < 0)
            return false;

        int width = -1;
        for (const auto &row : tables[u])
        {
            if (width < 0)
                width = static_cast<int>(row.size());
            if (static_cast<int>(row.size()) != width || tableKeys[u] > width)
                return false;
        }

        if (u == root)
            continue;
        const int p = parent[u];
        if (p < 0 || p >= n || p == u)
            return false;
        children[p].push_back(u);

        if (!tables[p].empty())
        {
            const int parentWidth = static_cast<int>(tables[p][0].size());
            if (joinColInParent[u] < 0 || joinColInParent[u] >= parentWidth)
                return false;
        }
        if (!tables[u].empty())
        {
            const int childWidth = static_cast<int>(tables[u][0].size());
            if (joinColInChild[u] < 0 || joinColInChild[u] >= childWidth)
                return false;
        }
    }

    topDownOrder.clear();
    std::queue<int> pending;
    pending.push(root);
    while (!pending.empty())
    {
        const int u = pending.front();
        pending.pop();
        topDownOrder.push_back(u);
        for (int c : children[u])
            pending.push(c);
    }
    return static_cast<int>(topDownOrder.size()) == n;
}

bool buildEdgeIndex(const Table &childTable,
                    int childJoinColumn,
                    const std::vector<Count> &childCounts,
                    EdgeIndex &index)
{
    index.clear();
    index.reserve(childTable.size());
    for (int rowId = 0; rowId < static_cast<int>(childTable.size()); ++rowId)
    {
        const Count count = childCounts[rowId];
        if (count == 0)
            continue;

        MatchBucket &bucket = index[childTable[rowId][childJoinColumn]];
        Count newTotal = 0;
        if (!checkedAdd(bucket.total, count, newTotal))
            return false;
        bucket.total = newTotal;
        bucket.tupleIds.push_back(rowId);
        bucket.cumulativeEnds.push_back(newTotal);
    }
    return true;
}

struct PositionState
{
    int tupleId = 0;
    Count localPosition = 0;
};

class CriticalPathMaterializer
{
public:
    CriticalPathMaterializer(const std::vector<Table> &tables,
                             const std::vector<std::vector<int>> &children,
                             const std::vector<int> &joinColInParent,
                             const std::vector<int> &tableKeys,
                             const std::vector<EdgeIndex> &edgeIndexes,
                             const std::vector<int> &columnOffsets,
                             Table &output)
        : tables_(tables),
          children_(children),
          joinColInParent_(joinColInParent),
          tableKeys_(tableKeys),
          edgeIndexes_(edgeIndexes),
          columnOffsets_(columnOffsets),
          output_(output)
    {
    }

    bool writeSubtree(int node,
                      const std::vector<PositionState> &states,
                      double &criticalMs) const
    {
        const double ownStart = sgxProfileNowMs();
        const int outputOffset = columnOffsets_[node];
#pragma omp parallel for schedule(static)
        for (int globalPosition = 0;
             globalPosition < static_cast<int>(states.size());
             ++globalPosition)
        {
            const auto &source = tables_[node][states[globalPosition].tupleId];
            auto &destination = output_[globalPosition];
            for (int c = 0; c < tableKeys_[node]; ++c)
                destination[outputOffset + c] = source[c];
        }
        const double ownMs = sgxProfileNowMs() - ownStart;

        double slowestChildMs = 0.0;
        const auto &nodeChildren = children_[node];
        for (int childIndex = 0;
             childIndex < static_cast<int>(nodeChildren.size());
             ++childIndex)
        {
            const int child = nodeChildren[childIndex];
            const double branchStart = sgxProfileNowMs();
            std::vector<PositionState> childStates(states.size());
            int invalid = 0;
#pragma omp parallel for schedule(static) reduction(| : invalid)
            for (int globalPosition = 0;
                 globalPosition < static_cast<int>(states.size());
                 ++globalPosition)
            {
                const PositionState state = states[globalPosition];
                const auto &source = tables_[node][state.tupleId];
                Count suffixStride = 1;
                for (int later = static_cast<int>(nodeChildren.size()) - 1;
                     later > childIndex;
                     --later)
                {
                    const int laterChild = nodeChildren[later];
                    const int laterKey = source[joinColInParent_[laterChild]];
                    const auto laterIt = edgeIndexes_[laterChild].find(laterKey);
                    Count nextStride = 0;
                    if (laterIt == edgeIndexes_[laterChild].end() ||
                        !checkedMultiply(suffixStride, laterIt->second.total, nextStride))
                    {
                        invalid = 1;
                        break;
                    }
                    suffixStride = nextStride;
                }
                if (invalid)
                    continue;

                const int joinKey = source[joinColInParent_[child]];
                const auto bucketIt = edgeIndexes_[child].find(joinKey);
                if (bucketIt == edgeIndexes_[child].end() || bucketIt->second.total <= 0)
                {
                    invalid = 1;
                    continue;
                }

                const MatchBucket &bucket = bucketIt->second;
                const Count childPosition =
                    (state.localPosition / suffixStride) % bucket.total;
                const auto endIt = std::upper_bound(bucket.cumulativeEnds.begin(),
                                                    bucket.cumulativeEnds.end(),
                                                    childPosition);
                const std::size_t selected =
                    static_cast<std::size_t>(endIt - bucket.cumulativeEnds.begin());
                if (selected >= bucket.tupleIds.size())
                {
                    invalid = 1;
                    continue;
                }
                const Count previousEnd =
                    selected == 0 ? 0 : bucket.cumulativeEnds[selected - 1];
                childStates[globalPosition] = {
                    bucket.tupleIds[selected], childPosition - previousEnd};
            }
            if (invalid)
                return false;

            const double branchMapMs = sgxProfileNowMs() - branchStart;
            double childCriticalMs = 0.0;
            if (!writeSubtree(child, childStates, childCriticalMs))
                return false;
            const double branchMs = branchMapMs + childCriticalMs;
            slowestChildMs = std::max(slowestChildMs, branchMs);
        }

        criticalMs = ownMs + slowestChildMs;
        return true;
    }

private:
    const std::vector<Table> &tables_;
    const std::vector<std::vector<int>> &children_;
    const std::vector<int> &joinColInParent_;
    const std::vector<int> &tableKeys_;
    const std::vector<EdgeIndex> &edgeIndexes_;
    const std::vector<int> &columnOffsets_;
    Table &output_;
};
} // namespace

NonObliviousJFYanResult NonObliviousJFYan(
    const std::vector<NonObliviousTable> &tables,
    const std::vector<int> &parent,
    int root,
    const std::vector<int> &joinColInParent,
    const std::vector<int> &joinColInChild,
    const std::vector<int> &tableKeys,
    int maxOutputRows)
{
    NonObliviousJFYanResult result;
    const double setupStart = sgxProfileNowMs();
    if (maxOutputRows < 0)
    {
        result.status = NonObliviousJFYanInvalidInput;
        return result;
    }

    std::vector<std::vector<int>> children;
    std::vector<int> topDownOrder;
    if (!validateInput(tables, parent, root, joinColInParent, joinColInChild,
                       tableKeys, children, topDownOrder))
    {
        result.status = NonObliviousJFYanInvalidInput;
        return result;
    }
    result.setupMs = sgxProfileNowMs() - setupStart;

    const int n = static_cast<int>(tables.size());
    std::vector<std::vector<Count>> contribution(n);
    std::vector<EdgeIndex> edgeIndexes(n); // Indexed by child node/edge.
    std::vector<double> bottomCriticalMs(n, 0.0);

    // Bottom-up contribution pass. A leaf tuple contributes one row. An
    // internal tuple contributes the product of the matching child totals.
    for (int orderPos = n - 1; orderPos >= 0; --orderPos)
    {
        const int node = topDownOrder[orderPos];
        const double nodeSetupStart = sgxProfileNowMs();
        contribution[node].assign(tables[node].size(), children[node].empty() ? 1 : 0);
        const double nodeSetupMs = sgxProfileNowMs() - nodeSetupStart;

        double slowestChildMs = 0.0;

        for (int child : children[node])
        {
            const double edgeStart = sgxProfileNowMs();
            if (!buildEdgeIndex(tables[child], joinColInChild[child],
                                contribution[child], edgeIndexes[child]))
            {
                result.status = NonObliviousJFYanCountOverflow;
                return result;
            }
            const double edgeMs = sgxProfileNowMs() - edgeStart;
            slowestChildMs = std::max(
                slowestChildMs, bottomCriticalMs[child] + edgeMs);
        }

        if (children[node].empty())
        {
            bottomCriticalMs[node] = nodeSetupMs;
            continue;
        }

        const double sharedStart = sgxProfileNowMs();
#pragma omp parallel for schedule(static)
        for (int rowId = 0; rowId < static_cast<int>(tables[node].size()); ++rowId)
        {
            Count count = 1;
            for (int child : children[node])
            {
                const int key = tables[node][rowId][joinColInParent[child]];
                const EdgeIndex &edgeIndex = edgeIndexes[child];
                const auto bucketIt = edgeIndex.find(key);
                if (bucketIt == edgeIndex.end())
                {
                    count = 0;
                    break;
                }
                Count next = 0;
                if (!checkedMultiply(count, bucketIt->second.total, next))
                {
                    count = -1;
                    break;
                }
                count = next;
            }
            contribution[node][rowId] = count;
        }

        for (Count count : contribution[node])
        {
            if (count < 0)
            {
                result.status = NonObliviousJFYanCountOverflow;
                return result;
            }
        }
        const double sharedMs = sgxProfileNowMs() - sharedStart;
        bottomCriticalMs[node] = nodeSetupMs + slowestChildMs + sharedMs;
    }
    result.bottomUpCriticalMs = bottomCriticalMs[root];

    const double materializeSetupStart = sgxProfileNowMs();
    std::vector<int> rootTupleIds;
    std::vector<Count> rootCumulativeEnds;
    Count exactRows = 0;
    for (int rowId = 0; rowId < static_cast<int>(tables[root].size()); ++rowId)
    {
        const Count count = contribution[root][rowId];
        if (count == 0)
            continue;
        Count newTotal = 0;
        if (!checkedAdd(exactRows, count, newTotal))
        {
            result.status = NonObliviousJFYanCountOverflow;
            return result;
        }
        exactRows = newTotal;
        rootTupleIds.push_back(rowId);
        rootCumulativeEnds.push_back(exactRows);
    }
    result.exactRows = exactRows;

    if (exactRows > maxOutputRows || exactRows > INT_MAX)
    {
        result.status = NonObliviousJFYanOutputTooLarge;
        return result;
    }

    std::vector<int> columnOffsets(n, 0);
    int totalColumns = 0;
    for (int node = 0; node < n; ++node)
    {
        columnOffsets[node] = totalColumns;
        if (tableKeys[node] > INT_MAX - totalColumns)
        {
            result.status = NonObliviousJFYanOutputTooLarge;
            return result;
        }
        totalColumns += tableKeys[node];
    }

    result.output.assign(static_cast<std::size_t>(exactRows), std::vector<int>(totalColumns, 0));
    const double materializeSetupMs = sgxProfileNowMs() - materializeSetupStart;
    if (exactRows == 0)
    {
        result.materializeCriticalMs = materializeSetupMs;
        result.parallelEstimateMs = result.setupMs + result.bottomUpCriticalMs +
                                    result.materializeCriticalMs;
        return result;
    }

    const double rootMapStart = sgxProfileNowMs();
    std::vector<PositionState> rootStates(static_cast<std::size_t>(exactRows));
    int invalidRoot = 0;
#pragma omp parallel for schedule(static) reduction(| : invalidRoot)
    for (int globalPosition = 0; globalPosition < static_cast<int>(exactRows); ++globalPosition)
    {
        const auto endIt = std::upper_bound(rootCumulativeEnds.begin(),
                                            rootCumulativeEnds.end(),
                                            static_cast<Count>(globalPosition));
        const std::size_t selected = static_cast<std::size_t>(endIt - rootCumulativeEnds.begin());
        if (selected >= rootTupleIds.size())
        {
            invalidRoot = 1;
            continue;
        }
        const Count previousEnd = selected == 0 ? 0 : rootCumulativeEnds[selected - 1];
        rootStates[globalPosition] = {
            rootTupleIds[selected],
            static_cast<Count>(globalPosition) - previousEnd};
    }
    if (invalidRoot)
    {
        result.status = NonObliviousJFYanInvalidInput;
        return result;
    }
    const double rootMapMs = sgxProfileNowMs() - rootMapStart;

    CriticalPathMaterializer materializer(tables, children, joinColInParent,
                                           tableKeys, edgeIndexes, columnOffsets,
                                           result.output);
    double treeMaterializeCriticalMs = 0.0;
    if (!materializer.writeSubtree(root, rootStates, treeMaterializeCriticalMs))
    {
        result.status = NonObliviousJFYanCountOverflow;
        return result;
    }
    result.materializeCriticalMs =
        materializeSetupMs + rootMapMs + treeMaterializeCriticalMs;
    result.parallelEstimateMs = result.setupMs + result.bottomUpCriticalMs +
                                result.materializeCriticalMs;

    return result;
}
