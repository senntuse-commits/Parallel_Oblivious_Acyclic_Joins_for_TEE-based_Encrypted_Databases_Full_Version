#include "Enclave_t.h"
#include "user_types.h"

#include <omp.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "BottomupSemiJoin.h"
#include "PrimitiveProfile.h"
#include "ObliYan.h"
#include "ParYan.h"
#include "JFYanDown.h"
#include "NonObliviousJFYan.h"
#include "MemoryProfile.h"
#include "sgx_profile.h"

using namespace std;

bool g_sgxProfileEnabled = false;
bool g_sgxProfileTimingEnabled = false;

namespace
{
#ifndef JFYAN_MAX_MATERIALIZED_ROWS
#define JFYAN_MAX_MATERIALIZED_ROWS 200000000
#endif
constexpr long long MaxPublicTableRows = (1LL << 28) - 1;
constexpr long long MaxMaterializedRows = JFYAN_MAX_MATERIALIZED_ROWS;
static_assert(MaxMaterializedRows > 0 && MaxMaterializedRows <= MaxPublicTableRows,
              "Invalid public materialization limit");

enum JoinMode
{
    ModeOurs = 0,
    ModeParYan = 1,
    ModeObliYan = 2,
    ModeNonObliviousJFYan = 3
};

// PublicTarget consumes an externally released Lambda without generating DP noise.
enum MaterializationPolicy
{
    ExactSizeBenchmark = 0,
    PublicTarget = 1,
    BenchmarkTargetSimulation = 2
};

enum ProfileStage
{
    StageRestoreTables = 0,
    StageRestoreMeta = 1,
    StageSetup = 2,
    StageCore1 = 3,
    StageCore2 = 4,
    StageSummarize = 5,
    StageInsideTotal = 6,
    StageDoExactRows = 7,
    StageDoSensitivity = 8,
    StageDoProtectedRows = 9,
    StageOursUpFilter = 10,
    StageOursRootExpand = 11,
    StageJFYanDown = 12,
    StageParYanUpFilter = 13,
    StageParYanDownFilter = 14,
    StageParYanJoin = 15,
    StagePrimitiveSort = 16,
    StagePrimitiveExpand = 17,
    StagePrimitiveCompact = 18,
    StagePrimitiveAggtree = 19,
    StageObliYanUpFilter = 20,
    StageObliYanDownFilter = 21,
    StageObliYanJoin = 22,
    StagePrimitivePhaseBase = 23,
    PrimitiveKindCount = 4,
    PrimitivePhaseCount = 9,
    StageMemoryStartLive = StagePrimitivePhaseBase + PrimitiveKindCount * PrimitivePhaseCount,
    StageMemoryPeakLive,
    StageMemoryEndLive,
    StageMemoryTotalAllocated,
    StageMemoryPositionAllocated,
    StageMemoryPositionPeakLive,
    StageMemoryPositionPeakIncrease,
    StageMemoryCopyTables,
    StageMemoryFinalOutput,
    StageMemoryBranchConcurrentUpperBound,
    StageMemoryAvailable,
    StageCount
};

void logLine(const char *s)
{
    ocall_print(s);
}

sgx_status_t reportJoinException(const char *where,
                                 const char *kind,
                                 const char *detail,
                                 sgx_status_t status)
{
    char buf[384];
    snprintf(buf, sizeof(buf),
             "[join error] %s: %s%s%s",
             where,
             kind,
             (detail && detail[0]) ? ": " : "",
             (detail && detail[0]) ? detail : "");
    logLine(buf);
    return status;
}

double nowMs()
{
    double ms = 0.0;
    ocall_now_ms(&ms);
    return ms;
}

void setStage(double *stageMs, int stageCount, int idx, double value)
{
    if (stageMs && idx >= 0 && idx < stageCount)
        stageMs[idx] = value;
}

void setPrimitiveStages(double *stageMs, int stageCount)
{
    setStage(stageMs, stageCount, StagePrimitiveSort, getPrimitiveProfileMs(PrimitiveSort));
    setStage(stageMs, stageCount, StagePrimitiveExpand, getPrimitiveProfileMs(PrimitiveExpand));
    setStage(stageMs, stageCount, StagePrimitiveCompact, getPrimitiveProfileMs(PrimitiveCompact));
    setStage(stageMs, stageCount, StagePrimitiveAggtree, getPrimitiveProfileMs(PrimitiveAggtree));
    for (int phase = 0; phase < PrimitivePhaseCount; ++phase)
    {
        int profilePhase = PrimitivePhaseOursUpFilter + phase;
        for (int kind = 0; kind < PrimitiveKindCount; ++kind)
        {
            int stageIdx = StagePrimitivePhaseBase + phase * PrimitiveKindCount + kind;
            setStage(stageMs, stageCount, stageIdx,
                     getPrimitiveProfileMsForPhase(profilePhase, kind));
        }
    }
}

void setMemoryStages(double *stageMs, int stageCount, const MemoryProfileSnapshot &memory)
{
    setStage(stageMs, stageCount, StageMemoryStartLive, (double)memory.startLiveBytes);
    setStage(stageMs, stageCount, StageMemoryPeakLive, (double)memory.peakLiveBytes);
    setStage(stageMs, stageCount, StageMemoryEndLive, (double)memory.endLiveBytes);
    setStage(stageMs, stageCount, StageMemoryTotalAllocated, (double)memory.totalAllocatedBytes);
    setStage(stageMs, stageCount, StageMemoryPositionAllocated, (double)memory.positionAllocatedBytes);
    setStage(stageMs, stageCount, StageMemoryPositionPeakLive, (double)memory.positionPeakLiveBytes);
    setStage(stageMs, stageCount, StageMemoryPositionPeakIncrease, (double)memory.positionPeakIncreaseBytes);
    setStage(stageMs, stageCount, StageMemoryCopyTables, (double)memory.copyTableBytes);
    setStage(stageMs, stageCount, StageMemoryFinalOutput, (double)memory.finalOutputBytes);
    setStage(stageMs, stageCount, StageMemoryBranchConcurrentUpperBound,
             (double)memory.branchConcurrentUpperBoundBytes);
    setStage(stageMs, stageCount, StageMemoryAvailable, memory.available ? 1.0 : 0.0);
}

void configureThreads(int threads)
{
    omp_set_dynamic(0);
    if (threads > 0)
        omp_set_num_threads(threads);
    // Enable inner parallel sorts within the outer JFYanDown tasks.
    // SGX OpenMP can otherwise serialize the inner regions.
    omp_set_max_active_levels(2);
    omp_set_nested(1);
}

sgx_status_t selectMaterializationTarget(
    const vector<Table> &tables, const vector<int> &parent, int root,
    const vector<int> &joinParent, const vector<int> &joinChild,
    int policy, int explicitTarget, double epsilon, double delta,
    int &target, double *stages = nullptr, int stageCount = 0)
{
    // Bound public row dimensions before int-indexed merge/sort workspaces are
    // formed. Two such tables plus dummy headers round up to at most 2^29 rows.
    for (const Table &table : tables)
        if (table.size() > static_cast<size_t>(MaxPublicTableRows))
            return SGX_ERROR_INVALID_PARAMETER;
    // These ABI slots are reserved, not private-cardinality output channels.
    setStage(stages, stageCount, StageDoExactRows, -1.0);
    setStage(stages, stageCount, StageDoSensitivity, -1.0);
    long long requested = 0;
    const char *source = nullptr;
    if (policy == PublicTarget)
    {
        if (explicitTarget <= 0)
        {
            logLine("Public-target execution requires a previously released positive Lambda; no DP generator is implemented.");
            return SGX_ERROR_INVALID_PARAMETER;
        }
        requested = explicitTarget;
        source = "externally-released-public-target";
    }
    else if (policy == BenchmarkTargetSimulation)
    {
        if (explicitTarget != 0 || !std::isfinite(epsilon) || !std::isfinite(delta) ||
            epsilon <= 0.0 || delta <= 0.0 || delta >= 1.0)
            return SGX_ERROR_INVALID_PARAMETER;
        // Deterministic benchmark sizing, not a DP release.
        const bool detailedProfile = sgxProfileEnabled();
        const bool timingProfile = g_sgxProfileTimingEnabled;
        setSgxProfileEnabled(false);
        const DOPaddingStats stats = computeDOPaddingStats(
            tables, parent, root, joinParent, joinChild, 0, epsilon, delta);
        setSgxProfileEnabled(detailedProfile);
        setSgxProfileTimingEnabled(timingProfile);
        if (stats.rrsTooExpensive || !stats.rrsAvailable)
        {
            logLine("Benchmark target-size simulation unavailable for this input/configuration; no target was released.");
            return SGX_ERROR_UNEXPECTED;
        }
        requested = stats.protectedRowsExact;
        source = "benchmark-target-size-simulation-NOT-DP";
    }
    else if (policy == ExactSizeBenchmark)
    {
        if (explicitTarget < 0)
            return SGX_ERROR_INVALID_PARAMETER;
        requested = explicitTarget > 0 ? explicitTarget :
            computeAcyclicJoinOutputSize(tables, parent, root, joinParent, joinChild);
        requested = std::max(1LL, requested);
        source = "exact-size-benchmark-output-size-is-public";
    }
    else
        return SGX_ERROR_INVALID_PARAMETER;

    // Never substitute the private exact size when a requested target is large.
    if (requested < 1 || requested > MaxMaterializedRows)
    {
        logLine("Requested target exceeds the configured materialization limit; no exact-size fallback is performed.");
        return SGX_ERROR_OUT_OF_MEMORY;
    }
    target = (int)requested;
    setStage(stages, stageCount, StageDoProtectedRows, (double)target);
    char buf[192];
    snprintf(buf, sizeof(buf), "[materialization target] Lambda=%d source=%s", target, source);
    logLine(buf);
    return SGX_SUCCESS;
}

bool restoreTables(const int *tablesFlat,
                   int flatLen,
                   const int *tableOffsets,
                   const int *tableRows,
                   const int *tableCols,
                   int tableCount,
                   vector<Table> &tables)
{
    if (!tablesFlat || !tableOffsets || !tableRows || !tableCols || tableCount <= 0 || flatLen < 0)
        return false;

    tables.clear();
    tables.resize(tableCount);

    for (int t = 0; t < tableCount; ++t)
    {
        int rows = tableRows[t];
        int cols = tableCols[t];
        int offset = tableOffsets[t];
        if (rows < 0 || cols < 0 || offset < 0)
            return false;

        long long cellCount = (long long)rows * (long long)cols;
        if (cellCount < 0 || offset + cellCount > flatLen)
            return false;

        Table table(rows, vector<int>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                table[i][j] = tablesFlat[offset + i * cols + j];
            }
        }
        tables[t] = std::move(table);
    }
    return true;
}

bool restoreVector(const int *src, int count, vector<int> &dst)
{
    if (!src || count <= 0)
        return false;
    dst.assign(src, src + count);
    return true;
}

sgx_status_t copyResult(const Table &res, int *result, int maxSize, int *retRows, int *retCols)
{
    if (!result || !retRows || !retCols || maxSize < 0)
        return SGX_ERROR_INVALID_PARAMETER;

    *retRows = (int)res.size();
    *retCols = res.empty() ? 0 : (int)res[0].size();

    long long required = (long long)(*retRows) * (long long)(*retCols);
    if (required > maxSize)
    {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "Enclave result buffer too small. required=%lld, max=%d",
                 required, maxSize);
        logLine(buf);
        return SGX_ERROR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < *retRows; ++i)
    {
        if ((int)res[i].size() != *retCols)
            return SGX_ERROR_UNEXPECTED;
        for (int j = 0; j < *retCols; ++j)
            result[i * (*retCols) + j] = res[i][j];
    }

    return SGX_SUCCESS;
}

sgx_status_t summarizeResult(const Table &res, int *retRows, int *retCols,
                            int policy, int publicTarget, const int *tableKeys, int tableCount)
{
    if (!retRows || !retCols)
        return SGX_ERROR_INVALID_PARAMETER;

    if (policy == PublicTarget)
    {
        // Return public dimensions only; keep empty-output status and validity private.
        if (publicTarget <= 0 || !tableKeys || tableCount <= 0)
            return SGX_ERROR_INVALID_PARAMETER;
        long long columns = 0;
        for (int i = 0; i < tableCount; ++i)
        {
            if (tableKeys[i] < 0)
                return SGX_ERROR_INVALID_PARAMETER;
            columns += tableKeys[i];
        }
        if (columns > INT_MAX)
            return SGX_ERROR_INVALID_PARAMETER;
        *retRows = publicTarget;
        *retCols = (int)columns;
        return SGX_SUCCESS;
    }

    *retRows = (int)res.size();
    *retCols = res.empty() ? 0 : (int)res[0].size();
    return SGX_SUCCESS;
}

sgx_status_t runJoin(int threads,
                     int mode,
                     int *tables_flat,
                     int flat_len,
                     int *table_offsets,
                     int *table_rows,
                     int *table_cols,
                     int table_count,
                     int *parent,
                     int root,
                     int *join_col_parent,
                     int *join_col_child,
                     int *table_keys,
                     int tau_or_out_size,
                     int materialize_padding,
                     int memory_profile,
                     double do_epsilon,
                     double do_delta,
                     Table &joinResult)
{
    try
    {
    (void)memory_profile;
    configureThreads(threads);

    if (table_count <= 0 || root < 0 || root >= table_count)
        return SGX_ERROR_INVALID_PARAMETER;

    vector<Table> tables;
    vector<int> parentVec;
    vector<int> joinParentVec;
    vector<int> joinChildVec;
    vector<int> tableKeysVec;

    if (!restoreTables(tables_flat, flat_len, table_offsets, table_rows, table_cols, table_count, tables) ||
        !restoreVector(parent, table_count, parentVec) ||
        !restoreVector(join_col_parent, table_count, joinParentVec) ||
        !restoreVector(join_col_child, table_count, joinChildVec) ||
        !restoreVector(table_keys, table_count, tableKeysVec))
    {
        return SGX_ERROR_INVALID_PARAMETER;
    }

    joinResult.clear();
    int materializedOutSize = 1;
    if (materialize_padding == PublicTarget && mode == ModeNonObliviousJFYan)
        return SGX_ERROR_INVALID_PARAMETER;
    sgx_status_t sizingStatus = selectMaterializationTarget(
        tables, parentVec, root, joinParentVec, joinChildVec,
        materialize_padding, tau_or_out_size, do_epsilon, do_delta, materializedOutSize);
    if (sizingStatus != SGX_SUCCESS)
        return sizingStatus;
    if (mode == ModeOurs)
    {
        logLine("Starting acyclic join: bottom-up + top-down");
        vector<Table> working = std::move(tables);
        setPrimitiveProfilePhase(PrimitivePhaseOursUpFilter);
        working[root] = bottomUpSemiJoin(working, parentVec, root, joinParentVec, joinChildVec, materializedOutSize, tableKeysVec[root]);
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        setPrimitiveProfilePhase(PrimitivePhaseJFYanDown);
        if (!working[root].empty())
            joinResult = JFYanDown(working, parentVec, root, joinParentVec, joinChildVec, tableKeysVec);
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        logLine("Acyclic join completed.");
    }
    else if (mode == ModeParYan)
    {
        logLine("Starting acyclic join: ParYanFilter + ParYanJoin");
        vector<Table> working = std::move(tables);
        auto filtered = ParYanFilter(working, parentVec, root, joinParentVec, joinChildVec, tableKeysVec);
        int outSize = materialize_padding == PublicTarget ? materializedOutSize :
            std::max(materializedOutSize, (int)std::max<long long>(filtered.second, 1));
        setPrimitiveProfilePhase(PrimitivePhaseParYanJoin);
        joinResult = ParYanJoin(filtered.first, parentVec, root, joinParentVec, joinChildVec, tableKeysVec, outSize);
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        logLine("ParYan acyclic join completed.");
    }
    else if (mode == ModeObliYan)
    {
        logLine("Starting acyclic join: ObliYan");
        joinResult = ObliYan(std::move(tables), parentVec, root, joinParentVec, joinChildVec, materializedOutSize);
        logLine("ObliYan completed.");
    }
    else if (mode == ModeNonObliviousJFYan)
    {
        logLine("Starting acyclic join: non-oblivious JFYan");
        NonObliviousJFYanResult nonOblivious = NonObliviousJFYan(
            tables, parentVec, root, joinParentVec, joinChildVec, tableKeysVec);
        if (nonOblivious.status == NonObliviousJFYanInvalidInput)
            return SGX_ERROR_INVALID_PARAMETER;
        if (nonOblivious.status == NonObliviousJFYanOutputTooLarge)
            return SGX_ERROR_OUT_OF_MEMORY;
        if (nonOblivious.status != NonObliviousJFYanOk)
            return SGX_ERROR_UNEXPECTED;
        joinResult = std::move(nonOblivious.output);
        logLine("Non-oblivious JFYan completed.");
    }
    else
    {
        return SGX_ERROR_INVALID_PARAMETER;
    }

    return SGX_SUCCESS;
    }
    catch (const length_error &e)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        return reportJoinException("runJoin", "size overflow", materialize_padding == PublicTarget ? nullptr : e.what(), SGX_ERROR_OUT_OF_MEMORY);
    }
    catch (const bad_alloc &e)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        return reportJoinException("runJoin", "allocation failed", materialize_padding == PublicTarget ? nullptr : e.what(), SGX_ERROR_OUT_OF_MEMORY);
    }
    catch (const exception &e)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        return reportJoinException("runJoin", "exception", materialize_padding == PublicTarget ? nullptr : e.what(), SGX_ERROR_UNEXPECTED);
    }
    catch (...)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        return reportJoinException("runJoin", "unknown exception", nullptr, SGX_ERROR_UNEXPECTED);
    }
}

sgx_status_t runJoinProfile(int threads,
                            int mode,
                            int *tables_flat,
                            int flat_len,
                            int *table_offsets,
                            int *table_rows,
                            int *table_cols,
                            int table_count,
                            int *parent,
                            int root,
                            int *join_col_parent,
                            int *join_col_child,
                            int *table_keys,
                            int tau_or_out_size,
                            int materialize_padding,
                            int memory_profile,
                            double do_epsilon,
                            double do_delta,
                            Table &joinResult,
                            double *stageMs,
                            int stageCount)
{
    try
    {
    configureThreads(threads);

    if (stageMs)
        for (int i = 0; i < stageCount; ++i)
            stageMs[i] = 0.0;

    double totalStart = nowMs();

    if (table_count <= 0 || root < 0 || root >= table_count)
        return SGX_ERROR_INVALID_PARAMETER;

    vector<Table> tables;
    vector<int> parentVec;
    vector<int> joinParentVec;
    vector<int> joinChildVec;
    vector<int> tableKeysVec;

    double t0 = nowMs();
    if (!restoreTables(tables_flat, flat_len, table_offsets, table_rows, table_cols, table_count, tables))
        return SGX_ERROR_INVALID_PARAMETER;
    double t1 = nowMs();
    setStage(stageMs, stageCount, StageRestoreTables, t1 - t0);

    if (!restoreVector(parent, table_count, parentVec) ||
        !restoreVector(join_col_parent, table_count, joinParentVec) ||
        !restoreVector(join_col_child, table_count, joinChildVec) ||
        !restoreVector(table_keys, table_count, tableKeysVec))
    {
        return SGX_ERROR_INVALID_PARAMETER;
    }
    double t2 = nowMs();
    setStage(stageMs, stageCount, StageRestoreMeta, t2 - t1);

    joinResult.clear();
    int materializedOutSize = 1;
    if (materialize_padding == PublicTarget && mode == ModeNonObliviousJFYan)
        return SGX_ERROR_INVALID_PARAMETER;
    sgx_status_t sizingStatus = selectMaterializationTarget(
        tables, parentVec, root, joinParentVec, joinChildVec,
        materialize_padding, tau_or_out_size, do_epsilon, do_delta,
        materializedOutSize, stageMs, stageCount);
    if (sizingStatus != SGX_SUCCESS)
        return sizingStatus;
    double afterPaddingStats = nowMs();
    memoryProfileBeginRun(memory_profile != 0 && materialize_padding != PublicTarget);
    resetPrimitiveProfile();
    if (mode == ModeOurs)
    {
        logLine("Starting acyclic join: bottom-up + top-down");
        vector<Table> working = std::move(tables);
        double t3 = afterPaddingStats;
        setStage(stageMs, stageCount, StageSetup, 0.0);

        setPrimitiveProfilePhase(PrimitivePhaseOursUpFilter);
        working[root] = bottomUpSemiJoin(working, parentVec, root, joinParentVec, joinChildVec, materializedOutSize, tableKeysVec[root]);
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        double t4 = nowMs();
        double bottomUpTotal = t4 - t3;
        double rootExpand = getLastBottomUpRootExpandMs();
        if (rootExpand < 0.0 || rootExpand > bottomUpTotal)
            rootExpand = 0.0;
        // Split measured bottom-up wall time into filtering and root expansion.
        setStage(stageMs, stageCount, StageCore1, bottomUpTotal);
        setStage(stageMs, stageCount, StageOursUpFilter, bottomUpTotal - rootExpand);
        setStage(stageMs, stageCount, StageOursRootExpand, rootExpand);

        setPrimitiveProfilePhase(PrimitivePhaseJFYanDown);
        if (!working[root].empty())
            joinResult = JFYanDown(working, parentVec, root, joinParentVec, joinChildVec, tableKeysVec);
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        double t5 = nowMs();
        // Report the full JFYanDown wall time.
        setStage(stageMs, stageCount, StageCore2, t5 - t4);
        setStage(stageMs, stageCount, StageJFYanDown, t5 - t4);
        logLine("Acyclic join completed.");
    }
    else if (mode == ModeParYan)
    {
        logLine("Starting acyclic join: ParYanFilter + ParYanJoin");
        vector<Table> working = std::move(tables);
        double t3 = afterPaddingStats;
        setStage(stageMs, stageCount, StageSetup, 0.0);

        auto filtered = ParYanFilter(working, parentVec, root, joinParentVec, joinChildVec, tableKeysVec);
        double t4 = nowMs();
        setStage(stageMs, stageCount, StageCore1, t4 - t3);
        setStage(stageMs, stageCount, StageParYanUpFilter, getLastParYanUpFilterMs());
        setStage(stageMs, stageCount, StageParYanDownFilter, getLastParYanDownFilterMs());

        int outSize = materialize_padding == PublicTarget ? materializedOutSize :
            std::max(materializedOutSize, (int)std::max<long long>(filtered.second, 1));
        setPrimitiveProfilePhase(PrimitivePhaseParYanJoin);
        joinResult = ParYanJoin(filtered.first, parentVec, root, joinParentVec, joinChildVec, tableKeysVec, outSize);
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        double t5 = nowMs();
        setStage(stageMs, stageCount, StageCore2, t5 - t4);
        setStage(stageMs, stageCount, StageParYanJoin, t5 - t4);
        logLine("ParYan acyclic join completed.");
    }
    else if (mode == ModeObliYan)
    {
        logLine("Starting acyclic join: ObliYan");
        double t3 = afterPaddingStats;
        setStage(stageMs, stageCount, StageSetup, 0.0);

        joinResult = ObliYan(std::move(tables), parentVec, root, joinParentVec, joinChildVec, materializedOutSize);
        double t4 = nowMs();
        setStage(stageMs, stageCount, StageCore1, t4 - t3);
        setStage(stageMs, stageCount, StageObliYanUpFilter, getLastObliYanUpFilterMs());
        setStage(stageMs, stageCount, StageObliYanDownFilter, getLastObliYanDownFilterMs());
        setStage(stageMs, stageCount, StageObliYanJoin, getLastObliYanJoinMs());
        logLine("ObliYan completed.");
    }
    else if (mode == ModeNonObliviousJFYan)
    {
        logLine("Starting acyclic join: non-oblivious JFYan");
        double t3 = afterPaddingStats;
        setStage(stageMs, stageCount, StageSetup, 0.0);

        NonObliviousJFYanResult nonOblivious = NonObliviousJFYan(
            tables, parentVec, root, joinParentVec, joinChildVec, tableKeysVec);
        if (nonOblivious.status == NonObliviousJFYanInvalidInput)
        {
            (void)memoryProfileEndRun();
            return SGX_ERROR_INVALID_PARAMETER;
        }
        if (nonOblivious.status == NonObliviousJFYanOutputTooLarge)
        {
            (void)memoryProfileEndRun();
            return SGX_ERROR_OUT_OF_MEMORY;
        }
        if (nonOblivious.status != NonObliviousJFYanOk)
        {
            (void)memoryProfileEndRun();
            return SGX_ERROR_UNEXPECTED;
        }
        joinResult = std::move(nonOblivious.output);

        double t4 = nowMs();
        char nonObliTiming[320];
        snprintf(nonObliTiming, sizeof(nonObliTiming),
                 "  [NonObliJFYan hypothetical-parallel-estimate NOT-measured-runtime] setup=%.2f ms bottomUpCritical=%.2f ms materializeCritical=%.2f ms total=%.2f ms measuredWall=%.2f ms",
                 nonOblivious.setupMs,
                 nonOblivious.bottomUpCriticalMs,
                 nonOblivious.materializeCriticalMs,
                 nonOblivious.parallelEstimateMs,
                 t4 - t3);
        logLine(nonObliTiming);
        // Report elapsed wall time; the critical-path estimate is log-only.
        setStage(stageMs, stageCount, StageCore1, t4 - t3);
        setStage(stageMs, stageCount, StageCore2, 0.0);
        // No comparable measured phase timings are available for this baseline.
        logLine("Non-oblivious JFYan completed.");
    }
    else
    {
        (void)memoryProfileEndRun();
        return SGX_ERROR_INVALID_PARAMETER;
    }

    double beforeSummary = nowMs();
    (void)beforeSummary;
    double totalEnd = nowMs();
    MemoryProfileSnapshot memory = memoryProfileEndRun();
    setPrimitiveStages(stageMs, stageCount);
    setMemoryStages(stageMs, stageCount, memory);
    setStage(stageMs, stageCount, StageSummarize, totalEnd - beforeSummary);
    setStage(stageMs, stageCount, StageInsideTotal, totalEnd - totalStart);
    return SGX_SUCCESS;
    }
    catch (const length_error &e)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        (void)memoryProfileEndRun();
        return reportJoinException("runJoinProfile", "size overflow", materialize_padding == PublicTarget ? nullptr : e.what(), SGX_ERROR_OUT_OF_MEMORY);
    }
    catch (const bad_alloc &e)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        (void)memoryProfileEndRun();
        return reportJoinException("runJoinProfile", "allocation failed", materialize_padding == PublicTarget ? nullptr : e.what(), SGX_ERROR_OUT_OF_MEMORY);
    }
    catch (const exception &e)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        (void)memoryProfileEndRun();
        return reportJoinException("runJoinProfile", "exception", materialize_padding == PublicTarget ? nullptr : e.what(), SGX_ERROR_UNEXPECTED);
    }
    catch (...)
    {
        setPrimitiveProfilePhase(PrimitivePhaseUnscoped);
        (void)memoryProfileEndRun();
        return reportJoinException("runJoinProfile", "unknown exception", nullptr, SGX_ERROR_UNEXPECTED);
    }
}
}

sgx_status_t empty_ecall(int threads)
{
    configureThreads(threads);
    return SGX_SUCCESS;
}

sgx_status_t AcyclicJoinRun(int threads,
                            int mode,
                            int *tables_flat,
                            int flat_len,
                            int *table_offsets,
                            int *table_rows,
                            int *table_cols,
                            int table_count,
                            int *parent,
                            int root,
                            int *join_col_parent,
                            int *join_col_child,
                            int *table_keys,
                            int tau_or_out_size,
                            int materialize_padding,
                            int memory_profile,
                            double do_epsilon,
                            double do_delta,
                            int *result,
                            int max_size,
                            int *ret_rows,
                            int *ret_cols)
{
    setSgxProfileEnabled(false);
    setSgxProfileTimingEnabled(false);
    if (ret_rows)
        *ret_rows = 0;
    if (ret_cols)
        *ret_cols = 0;

    // Raw plaintext copy-out is a research/debug facility, not the public-target
    // interface. A deployment needs an authenticated encrypted client channel.
    if (materialize_padding == PublicTarget)
        return SGX_ERROR_INVALID_PARAMETER;

    Table joinResult;
    sgx_status_t status = runJoin(threads, mode, tables_flat, flat_len, table_offsets,
                                  table_rows, table_cols, table_count, parent, root,
                                  join_col_parent, join_col_child, table_keys,
                                  tau_or_out_size, materialize_padding, memory_profile,
                                  do_epsilon, do_delta, joinResult);
    if (status != SGX_SUCCESS)
        return status;

    return copyResult(joinResult, result, max_size, ret_rows, ret_cols);
}

sgx_status_t AcyclicJoinRunSummary(int threads,
                                   int mode,
                                   int *tables_flat,
                                   int flat_len,
                                   int *table_offsets,
                                   int *table_rows,
                                   int *table_cols,
                                   int table_count,
                                   int *parent,
                                   int root,
                                   int *join_col_parent,
                                   int *join_col_child,
                                   int *table_keys,
                                   int tau_or_out_size,
                                   int materialize_padding,
                                   int memory_profile,
                                   double do_epsilon,
                                   double do_delta,
                                   int *ret_rows,
                                   int *ret_cols)
{
    setSgxProfileEnabled(false);
    setSgxProfileTimingEnabled(false);
    if (ret_rows)
        *ret_rows = 0;
    if (ret_cols)
        *ret_cols = 0;

    Table joinResult;
    sgx_status_t status = runJoin(threads, mode, tables_flat, flat_len, table_offsets,
                                  table_rows, table_cols, table_count, parent, root,
                                  join_col_parent, join_col_child, table_keys,
                                  tau_or_out_size, materialize_padding, memory_profile,
                                  do_epsilon, do_delta, joinResult);
    if (status != SGX_SUCCESS)
        return status;

    return summarizeResult(joinResult, ret_rows, ret_cols, materialize_padding,
                           tau_or_out_size, table_keys, table_count);
}

sgx_status_t AcyclicJoinRunProfile(int threads,
                                   int mode,
                                   int *tables_flat,
                                   int flat_len,
                                   int *table_offsets,
                                   int *table_rows,
                                   int *table_cols,
                                   int table_count,
                                   int *parent,
                                   int root,
                                   int *join_col_parent,
                                   int *join_col_child,
                                   int *table_keys,
                                   int tau_or_out_size,
                                   int materialize_padding,
                                   int memory_profile,
                                   double do_epsilon,
                                   double do_delta,
                                   int *ret_rows,
                                   int *ret_cols,
                                   double *stage_ms,
                                   int stage_count)
{
    setSgxProfileEnabled(materialize_padding != PublicTarget);
    if (ret_rows)
        *ret_rows = 0;
    if (ret_cols)
        *ret_cols = 0;

    Table joinResult;
    sgx_status_t status = runJoinProfile(threads, mode, tables_flat, flat_len, table_offsets,
                                         table_rows, table_cols, table_count, parent, root,
                                         join_col_parent, join_col_child, table_keys,
                                         tau_or_out_size, materialize_padding, memory_profile,
                                         do_epsilon, do_delta, joinResult, stage_ms, stage_count);
    if (status != SGX_SUCCESS)
    {
        setSgxProfileEnabled(false);
        setSgxProfileTimingEnabled(false);
        return status;
    }

    double t0 = nowMs();
    status = summarizeResult(joinResult, ret_rows, ret_cols, materialize_padding,
                             tau_or_out_size, table_keys, table_count);
    double t1 = nowMs();
    setStage(stage_ms, stage_count, StageSummarize, t1 - t0);
    if (stage_ms && StageInsideTotal < stage_count)
        stage_ms[StageInsideTotal] += (t1 - t0);
    setSgxProfileEnabled(false);
    setSgxProfileTimingEnabled(false);
    return status;
}

sgx_status_t AcyclicJoinRunStageSummary(int threads,
                                        int mode,
                                        int *tables_flat,
                                        int flat_len,
                                        int *table_offsets,
                                        int *table_rows,
                                        int *table_cols,
                                        int table_count,
                                        int *parent,
                                        int root,
                                        int *join_col_parent,
                                        int *join_col_child,
                                        int *table_keys,
                                        int tau_or_out_size,
                                        int materialize_padding,
                                        int memory_profile,
                                        double do_epsilon,
                                        double do_delta,
                                        int *ret_rows,
                                        int *ret_cols,
                                        double *stage_ms,
                                        int stage_count)
{
    setSgxProfileEnabled(false);
    setSgxProfileTimingEnabled(true);
    if (ret_rows)
        *ret_rows = 0;
    if (ret_cols)
        *ret_cols = 0;

    Table joinResult;
    sgx_status_t status = runJoinProfile(threads, mode, tables_flat, flat_len, table_offsets,
                                         table_rows, table_cols, table_count, parent, root,
                                         join_col_parent, join_col_child, table_keys,
                                         tau_or_out_size, materialize_padding, memory_profile,
                                         do_epsilon, do_delta, joinResult, stage_ms, stage_count);
    if (status != SGX_SUCCESS)
    {
        setSgxProfileTimingEnabled(false);
        return status;
    }

    double t0 = nowMs();
    status = summarizeResult(joinResult, ret_rows, ret_cols, materialize_padding,
                             tau_or_out_size, table_keys, table_count);
    double t1 = nowMs();
    setStage(stage_ms, stage_count, StageSummarize, t1 - t0);
    if (stage_ms && StageInsideTotal < stage_count)
        stage_ms[StageInsideTotal] += (t1 - t0);
    setSgxProfileTimingEnabled(false);
    return status;
}
