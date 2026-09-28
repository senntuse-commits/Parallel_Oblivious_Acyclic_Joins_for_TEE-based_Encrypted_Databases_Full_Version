#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <malloc.h>
#include <unistd.h>

#include <sgx_urts.h>

#include "Enclave_u.h"
#include "util.h"

using namespace std;

namespace
{
constexpr int ModeOurs = 0;
constexpr int ModeParYan = 1;
constexpr int ModeObliYan = 2;
constexpr int ModeNonObliviousJFYan = 3;
constexpr int ExactSizeBenchmark = 0;
constexpr int PublicTarget = 1;
constexpr int BenchmarkTargetSimulation = 2;

// Stage indices must match Enclave/Enclave.cpp.
constexpr int StagePrimitivePhaseBase = 23;
constexpr int PrimitiveKindCount = 4;
constexpr int PrimitivePhaseCount = 9;
constexpr int StageMemoryStartLive = StagePrimitivePhaseBase + PrimitiveKindCount * PrimitivePhaseCount;
constexpr int StageMemoryPeakLive = StageMemoryStartLive + 1;
constexpr int StageMemoryEndLive = StageMemoryStartLive + 2;
constexpr int StageMemoryTotalAllocated = StageMemoryStartLive + 3;
constexpr int StageMemoryPositionAllocated = StageMemoryStartLive + 4;
constexpr int StageMemoryPositionPeakLive = StageMemoryStartLive + 5;
constexpr int StageMemoryPositionPeakIncrease = StageMemoryStartLive + 6;
constexpr int StageMemoryCopyTables = StageMemoryStartLive + 7;
constexpr int StageMemoryFinalOutput = StageMemoryStartLive + 8;
constexpr int StageMemoryBranchConcurrentUpperBound = StageMemoryStartLive + 9;
constexpr int StageMemoryAvailable = StageMemoryStartLive + 10;
constexpr int ProfileStageCount = StageMemoryAvailable + 1;
constexpr int StageOursUpFilter = 10;
constexpr int StageOursRootExpand = 11;
constexpr int StageJFYanDown = 12;
constexpr int StageParYanUpFilter = 13;
constexpr int StageParYanDownFilter = 14;
constexpr int StageParYanJoin = 15;
constexpr int StagePrimitiveSort = 16;
constexpr int StagePrimitiveExpand = 17;
constexpr int StagePrimitiveCompact = 18;
constexpr int StagePrimitiveAggtree = 19;
constexpr int StageObliYanUpFilter = 20;
constexpr int StageObliYanDownFilter = 21;
constexpr int StageObliYanJoin = 22;

int globalThreadCount = 16;

void die(const char *msg)
{
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

const char *modeName(int mode)
{
    switch (mode)
    {
    case ModeOurs:
        return "JFYan";
    case ModeParYan:
        return "ParYan";
    case ModeObliYan:
        return "ObliYan";
    case ModeNonObliviousJFYan:
        return "NonObliJFYan";
    default:
        return "unknown";
    }
}

const char *sgxStatusName(sgx_status_t status)
{
    switch (status)
    {
    case SGX_SUCCESS:
        return "SGX_SUCCESS";
    case SGX_ERROR_UNEXPECTED:
        return "SGX_ERROR_UNEXPECTED";
    case SGX_ERROR_INVALID_PARAMETER:
        return "SGX_ERROR_INVALID_PARAMETER";
    case SGX_ERROR_OUT_OF_MEMORY:
        return "SGX_ERROR_OUT_OF_MEMORY";
    case SGX_ERROR_ENCLAVE_LOST:
        return "SGX_ERROR_ENCLAVE_LOST";
    case SGX_ERROR_ENCLAVE_CRASHED:
        return "SGX_ERROR_ENCLAVE_CRASHED";
    case SGX_ERROR_OUT_OF_EPC:
        return "SGX_ERROR_OUT_OF_EPC";
    case SGX_ERROR_INVALID_ENCLAVE:
        return "SGX_ERROR_INVALID_ENCLAVE";
    case SGX_ERROR_INVALID_ENCLAVE_ID:
        return "SGX_ERROR_INVALID_ENCLAVE_ID";
    default:
        return "SGX_ERROR_UNKNOWN";
    }
}

const char *primitivePhaseName(int phase)
{
    static const char *names[PrimitivePhaseCount] = {
        "JFYanUp",
        "JFYanRootExpand",
        "JFYanDown",
        "ParYanUp",
        "ParYanDown",
        "ParYanJoin",
        "ObliYanUp",
        "ObliYanDown",
        "ObliYan",
    };
    return (phase >= 0 && phase < PrimitivePhaseCount) ? names[phase] : "unknownPhase";
}

const char *primitiveKindName(int kind)
{
    static const char *names[PrimitiveKindCount] = {
        "sort",
        "expand",
        "compact",
        "aggtree",
    };
    return (kind >= 0 && kind < PrimitiveKindCount) ? names[kind] : "unknownPrimitive";
}

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [-JFYan|-ParYan|-ObliYan|-NonObliJFYan|--all] [--bench-only|--profile|--stage-profile|--memory-profile] [--epc-page-profile] [-t threads] [--thread-sweep list] [--warmup-jfyan] [-m max_cells] [-tau value]\n"
              << "                  [--sql18 dir|--sql85 dir|--sql85-chain3 dir|--sql85-returns-star dir|--tpch9 dir|--job1a dir]\n"
              << "                  [--tree-workload projected_dir] [--print-limit rows]\n"
              << "Default: sample data, -JFYan -t 16 -m 1000000 --print-limit 20\n"
              << "--bench-only runs the join and returns only rows/cols, avoiding full result copy-out.\n"
              << "--profile also returns detailed enclave-side stage timings for debugging.\n"
              << "--stage-profile returns coarse stage timings without detailed enclave logs.\n"
              << "--memory-profile reports tracked C++ heap use; run through exec.sh so the profiler is compiled in.\n"
              << "--epc-page-profile counts SGX EPC page-out/page-in attempts around each ECALL; run as root and keep it separate from formal runtime.\n"
              << "--thread-sweep reuses one enclave and runs comma-separated thread counts, e.g. 8,16,24,32.\n"
              << "--warmup-jfyan runs one unreported JFYan warm-up in the same enclave before the requested measurements.\n";
    std::cout << "--public-target Lambda (alias: -tau with --materialize-padding) consumes an externally released public bound.\n";
    std::cout << "--materialize-padding requires that public bound; this program does not implement a DP target generator.\n";
    std::cout << "--benchmark-target-simulation explicitly enables historical deterministic target sizing (NOT a DP release).\n";
    std::cout << "--do-epsilon and --do-delta configure only that benchmark simulation.\n";
    std::cout << "--no-materialize-padding runs an exact-size experiment whose output size is public.\n";
    std::cout << "-NonObliJFYan runs the exact-output, non-oblivious JFYan baseline; it is not added to --all.\n";
    std::cout << "--tpch9 enables --materialize-padding by default; SQL18/SQL85 variants keep it off by default.\n";
}

std::vector<int> parseThreadList(const std::string &text)
{
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        int v = std::atoi(item.c_str());
        if (v > 0)
            values.push_back(v);
    }
    return values;
}

int parsePositiveTarget(const char *text)
{
    char *end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > INT_MAX)
        die("The materialization target must be an integer in [1, INT_MAX].");
    return static_cast<int>(value);
}

const char *stageName(int mode, int idx)
{
    static std::string dynamicName;
    if (idx >= StagePrimitivePhaseBase && idx < StageMemoryStartLive)
    {
        int off = idx - StagePrimitivePhaseBase;
        dynamicName = std::string(primitivePhaseName(off / PrimitiveKindCount)) +
                      "." + primitiveKindName(off % PrimitiveKindCount);
        return dynamicName.c_str();
    }

    switch (idx)
    {
    case 0:
        return "restoreTables";
    case 1:
        return "restoreMeta";
    case 2:
        return "setupCopy";
    case 3:
        return mode == ModeOurs ? "bottomUpSemiJoin" :
               mode == ModeParYan ? "ParYanFilter" :
               mode == ModeObliYan ? "ObliYan" : "NonObliviousJFYan";
    case 4:
        return mode == ModeOurs ? "JFYanDown" :
               mode == ModeParYan ? "ParYanJoin" : "unused";
    case 5:
        return "summarize";
    case 6:
        return "insideTotal";
    case 7:
        return "reservedPrivateRows";
    case 8:
        return "reservedPrivateSensitivity";
    case 9:
        return "publicMaterializationTarget";
    case StageOursUpFilter:
        return "oursUpFilter";
    case StageOursRootExpand:
        return "oursRootExpand";
    case StageJFYanDown:
        return "JFYanDown";
    case StageParYanUpFilter:
        return "ParYanUpFilter";
    case StageParYanDownFilter:
        return "ParYanDownFilter";
    case StageParYanJoin:
        return "ParYanJoin";
    case StagePrimitiveSort:
        return "primitiveSort";
    case StagePrimitiveExpand:
        return "primitiveExpand";
    case StagePrimitiveCompact:
        return "primitiveCompact";
    case StagePrimitiveAggtree:
        return "primitiveAggtree";
    case StageObliYanUpFilter:
        return "ObliYanUpFilter";
    case StageObliYanDownFilter:
        return "ObliYanDownFilter";
    case StageObliYanJoin:
        return "ObliYan";
    default:
        return "unknown";
    }
}

bool stageIsMetric(int idx)
{
    return idx >= 7 && idx <= 9;
}

bool stageIsMemoryMetric(int idx)
{
    return idx >= StageMemoryStartLive && idx <= StageMemoryAvailable;
}

std::vector<Table> sampleTables()
{
    Table r1 = {{15, 10, 3}, {8, 7, 5}, {9, 10, 1}, {8, 9, 3}};
    Table r2 = {{10, 8}, {11, 9}, {12, 8}, {13, 15}};
    Table r3 = {{7, 15}, {7, 16}, {10, 17}, {9, 18}};
    Table r4 = {{1, 1}, {3, 4}, {3, 2}, {5, 1}};
    Table r5 = {{1, 1}, {3, 2}, {3, 3}, {1, 4}};
    Table r6 = {{1, 5}, {2, 6}, {4, 7}, {4, 8}};
    return {r1, r2, r3, r4, r5, r6};
}

std::string pathJoin(const std::string &dir, const char *file)
{
    if (dir.empty())
        return file;
    char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\')
        return dir + file;
    return dir + "/" + file;
}

Table loadRequiredTable(const std::string &path, int expectedCols)
{
    Table table = loadData(path);
    if (table.empty())
    {
        std::string msg = "Cannot load table or table is empty: " + path;
        die(msg.c_str());
    }
    for (int i = 0; i < (int)table.size(); ++i)
    {
        if ((int)table[i].size() != expectedCols)
        {
            std::fprintf(stderr, "Bad column count in %s at row %d: got %d, expected %d\n",
                         path.c_str(), i, (int)table[i].size(), expectedCols);
            std::exit(1);
        }
    }
    return table;
}

std::vector<Table> loadSQL18Tables(const std::string &dir)
{
    return {
        loadRequiredTable(pathJoin(dir, "R1_catalog_sales.tbl"), 4),
        loadRequiredTable(pathJoin(dir, "R2_date_dim.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R3_item.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R4_cd1.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R5_customer.tbl"), 3),
        loadRequiredTable(pathJoin(dir, "R6_cd2.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R7_customer_address.tbl"), 2),
    };
}

std::vector<Table> loadSQL85Tables(const std::string &dir)
{
    return {
        loadRequiredTable(pathJoin(dir, "R1_web_sales.tbl"), 3),
        loadRequiredTable(pathJoin(dir, "R2_web_page.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R3_date_dim.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R4_web_returns.tbl"), 5),
        loadRequiredTable(pathJoin(dir, "R5_cd1.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R6_cd2.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R7_customer_address.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R8_reason.tbl"), 2),
    };
}

std::vector<Table> loadSQL85Chain3Tables(const std::string &dir)
{
    return {
        loadRequiredTable(pathJoin(dir, "R1_web_sales.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R2_web_returns.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R3_cd1.tbl"), 2),
    };
}

std::vector<Table> loadSQL85ReturnsStarTables(const std::string &dir)
{
    return {
        loadRequiredTable(pathJoin(dir, "R1_web_returns.tbl"), 4),
        loadRequiredTable(pathJoin(dir, "R2_cd1.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R3_cd2.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R4_customer_address.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R5_reason.tbl"), 2),
    };
}

static long long compositePairKey(int a, int b)
{
    return (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
}

static int compositePairId(std::unordered_map<long long, int> &ids, int a, int b)
{
    long long key = compositePairKey(a, b);
    auto it = ids.find(key);
    if (it != ids.end())
        return it->second;
    int id = (int)ids.size() + 1;
    ids.emplace(key, id);
    return id;
}

std::vector<Table> loadTPCH9Tables(const std::string &dir)
{
    Table orders = loadRequiredTable(pathJoin(dir, "orders_ok.txt"), 1);
    Table lineitemRaw = loadRequiredTable(pathJoin(dir, "lineitem_sk_pk_ok.txt"), 3);
    Table partsuppRaw = loadRequiredTable(pathJoin(dir, "partsupp.txt"), 2);
    Table part = loadRequiredTable(pathJoin(dir, "part.txt"), 1);
    Table supplier = loadRequiredTable(pathJoin(dir, "supplier_sk_nk.txt"), 2);
    Table nation = loadRequiredTable(pathJoin(dir, "nation.txt"), 1);

    std::unordered_map<long long, int> pairIds;
    pairIds.reserve((lineitemRaw.size() + partsuppRaw.size()) * 2 + 1);

    Table lineitem;
    lineitem.reserve(lineitemRaw.size());
    for (const auto &row : lineitemRaw)
    {
        int suppKey = row[0];
        int partKey = row[1];
        int orderKey = row[2];
        int pairKey = compositePairId(pairIds, suppKey, partKey);
        lineitem.push_back({orderKey, pairKey});
    }

    Table partsupp;
    partsupp.reserve(partsuppRaw.size());
    for (const auto &row : partsuppRaw)
    {
        int suppKey = row[0];
        int partKey = row[1];
        int pairKey = compositePairId(pairIds, suppKey, partKey);
        partsupp.push_back({pairKey, partKey, suppKey});
    }

    return {lineitem, orders, partsupp, part, supplier, nation};
}

long long loadExpectedRows(const std::string &dir)
{
    std::ifstream in(pathJoin(dir, "expected.txt"));
    long long expected = -1;
    if (in)
        in >> expected;
    return expected;
}

struct DatasetConfig
{
    std::vector<Table> tables;
    std::vector<int> parent;
    std::vector<int> tableKeys;
    std::vector<int> joinColInParent;
    std::vector<int> joinColInChild;
    int root = 0;
    long long expectedRows = -1;
};

DatasetConfig loadSQL18Dataset(const std::string &dir)
{
    DatasetConfig cfg;
    cfg.tables = loadSQL18Tables(dir);
    cfg.parent = {-1, 0, 0, 0, 0, 4, 4};
    cfg.tableKeys = {4, 2, 2, 2, 3, 2, 2};
    cfg.joinColInParent = {-1, 0, 1, 2, 3, 1, 2};
    cfg.joinColInChild = {-1, 0, 0, 0, 0, 0, 0};
    cfg.expectedRows = loadExpectedRows(dir);
    return cfg;
}

DatasetConfig loadSQL85Dataset(const std::string &dir)
{
    DatasetConfig cfg;
    cfg.tables = loadSQL85Tables(dir);
    cfg.parent = {-1, 0, 0, 0, 3, 3, 3, 3};
    cfg.tableKeys = {3, 2, 2, 5, 2, 2, 2, 2};
    cfg.joinColInParent = {-1, 0, 1, 2, 1, 2, 3, 4};
    cfg.joinColInChild = {-1, 0, 0, 0, 0, 0, 0, 0};
    cfg.expectedRows = loadExpectedRows(dir);
    return cfg;
}

DatasetConfig loadSQL85Chain3Dataset(const std::string &dir)
{
    DatasetConfig cfg;
    cfg.tables = loadSQL85Chain3Tables(dir);
    cfg.parent = {-1, 0, 1};
    cfg.tableKeys = {2, 2, 2};
    cfg.joinColInParent = {-1, 0, 1};
    cfg.joinColInChild = {-1, 0, 0};
    cfg.expectedRows = loadExpectedRows(dir);
    return cfg;
}

DatasetConfig loadSQL85ReturnsStarDataset(const std::string &dir)
{
    DatasetConfig cfg;
    cfg.tables = loadSQL85ReturnsStarTables(dir);
    cfg.parent = {-1, 0, 0, 0, 0};
    cfg.tableKeys = {4, 2, 2, 2, 2};
    cfg.joinColInParent = {-1, 0, 1, 2, 3};
    cfg.joinColInChild = {-1, 0, 0, 0, 0};
    cfg.expectedRows = loadExpectedRows(dir);
    return cfg;
}

DatasetConfig loadTPCH9Dataset(const std::string &dir)
{
    DatasetConfig cfg;
    cfg.tables = loadTPCH9Tables(dir);
    cfg.parent = {-1, 0, 0, 2, 2, 4};
    cfg.tableKeys = {2, 1, 3, 1, 2, 1};
    cfg.joinColInParent = {-1, 0, 1, 1, 2, 1};
    cfg.joinColInChild = {-1, 0, 0, 0, 0, 0};
    cfg.expectedRows = loadExpectedRows(dir);
    return cfg;
}

DatasetConfig loadJOB1ADataset(const std::string &dir)
{
    DatasetConfig cfg;
    // Join core of JOB Query 1a, rooted at movie_companies:
    // company_type <- movie_companies -> title
    //                                  -> movie_info_idx -> info_type
    cfg.tables = {
        loadRequiredTable(pathJoin(dir, "R1_movie_companies.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R2_company_type.tbl"), 1),
        loadRequiredTable(pathJoin(dir, "R3_title.tbl"), 1),
        loadRequiredTable(pathJoin(dir, "R4_movie_info_idx.tbl"), 2),
        loadRequiredTable(pathJoin(dir, "R5_info_type.tbl"), 1),
    };
    cfg.parent = {-1, 0, 0, 0, 3};
    cfg.tableKeys = {2, 1, 1, 2, 1};
    cfg.joinColInParent = {-1, 0, 1, 1, 1};
    cfg.joinColInChild = {-1, 0, 0, 0, 0};
    cfg.expectedRows = loadExpectedRows(dir);
    return cfg;
}

DatasetConfig loadTreeWorkloadDataset(const std::string &dir)
{
    DatasetConfig cfg;
    std::ifstream treeIn(pathJoin(dir, "tree.txt"));
    if (!treeIn)
        die(("Cannot open tree-workload metadata: " + pathJoin(dir, "tree.txt")).c_str());

    int id = 0;
    int parent = -1;
    int joinParent = -1;
    int joinChild = -1;
    int columns = 0;
    int rootCount = 0;
    while (treeIn >> id >> parent >> joinParent >> joinChild >> columns)
    {
        if (id != (int)cfg.parent.size() || columns <= 0)
            die("Invalid tree.txt: relation ids must be consecutive and column counts must be positive.");
        cfg.parent.push_back(parent);
        cfg.joinColInParent.push_back(joinParent);
        cfg.joinColInChild.push_back(joinChild);
        cfg.tableKeys.push_back(columns);
        if (parent == -1)
        {
            cfg.root = id;
            ++rootCount;
        }
    }
    if (cfg.parent.empty() || rootCount != 1)
        die("Invalid tree.txt: exactly one root is required.");

    const int relationCount = (int)cfg.parent.size();
    for (int relation = 0; relation < relationCount; ++relation)
    {
        if (relation == cfg.root)
        {
            if (cfg.joinColInParent[relation] != -1 || cfg.joinColInChild[relation] != -1)
                die("Invalid tree.txt: root join columns must be -1.");
            continue;
        }
        int p = cfg.parent[relation];
        if (p < 0 || p >= relationCount || p == relation ||
            cfg.joinColInParent[relation] < 0 ||
            cfg.joinColInParent[relation] >= cfg.tableKeys[p] ||
            cfg.joinColInChild[relation] < 0 ||
            cfg.joinColInChild[relation] >= cfg.tableKeys[relation])
            die("Invalid tree.txt parent or join column.");
    }

    cfg.tables.assign(relationCount, Table());
    std::ifstream dataIn(pathJoin(dir, "tables.tbl"));
    if (!dataIn)
        die(("Cannot open tree-workload data: " + pathJoin(dir, "tables.tbl")).c_str());

    std::string line;
    int lineNumber = 0;
    while (std::getline(dataIn, line))
    {
        ++lineNumber;
        if (line.empty())
            continue;
        std::istringstream rowIn(line);
        int relation = -1;
        rowIn >> relation;
        if (relation < 0 || relation >= relationCount)
        {
            std::fprintf(stderr, "Invalid relation id in tables.tbl at line %d\n", lineNumber);
            std::exit(1);
        }
        std::vector<int> row;
        int value = 0;
        while (rowIn >> value)
            row.push_back(value);
        if ((int)row.size() != cfg.tableKeys[relation])
        {
            std::fprintf(stderr,
                         "Bad column count in tables.tbl at line %d: relation %d got %d, expected %d\n",
                         lineNumber, relation, (int)row.size(), cfg.tableKeys[relation]);
            std::exit(1);
        }
        cfg.tables[relation].push_back(std::move(row));
    }
    for (int relation = 0; relation < relationCount; ++relation)
    {
        if (cfg.tables[relation].empty())
            die("Invalid tables.tbl: every relation must contain at least one row.");
    }

    cfg.expectedRows = loadExpectedRows(dir);
    if (cfg.expectedRows < 0)
        die("Missing expected.txt for tree workload.");
    return cfg;
}

struct RunSummary
{
    int mode = ModeOurs;
    int rows = 0;
    int cols = 0;
    double ms = 0.0;
    sgx_status_t callStatus = SGX_SUCCESS;
    sgx_status_t ecallStatus = SGX_SUCCESS;
    std::vector<double> stageMs;
    bool paddingStatsOnly = false;
    bool skipped = false;
    bool hasResultSignature = false;
    std::uint64_t resultSignature = 0;
    long long resultCellCount = 0;
    long long ksgxdSystemTicksBefore = -1;
    long long ksgxdSystemTicksAfter = -1;
    bool epcPageProfileAvailable = false;
    long long epcPageOutAttempts = -1;
    long long epcPageInAttempts = -1;
    long long epcReclaimBatches = -1;
};

constexpr const char *TraceRoot = "/sys/kernel/tracing";

bool writeTraceControl(const char *relativePath, const char *value)
{
    std::ofstream out(std::string(TraceRoot) + "/" + relativePath);
    if (!out)
        return false;
    out << value;
    return static_cast<bool>(out);
}

bool beginEpcPageProfile()
{
    if (!writeTraceControl("function_profile_enabled", "0\n") ||
        !writeTraceControl("current_tracer", "nop\n") ||
        !writeTraceControl("set_ftrace_filter",
                           "sgx_encl_ewb\nsgx_encl_eldu\nsgx_reclaim_pages\n") ||
        !writeTraceControl("current_tracer", "function\n"))
        return false;
    return writeTraceControl("function_profile_enabled", "1\n");
}

void endEpcPageProfile(RunSummary &summary)
{
    writeTraceControl("function_profile_enabled", "0\n");
    long long ewb = 0;
    long long eldu = 0;
    long long reclaim = 0;
    bool readAny = false;
    for (int cpu = 0; cpu < 1024; ++cpu)
    {
        std::ifstream in(std::string(TraceRoot) + "/trace_stat/function" +
                         std::to_string(cpu));
        if (!in)
            continue;
        readAny = true;
        std::string name;
        long long count = 0;
        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream fields(line);
            if (!(fields >> name >> count))
                continue;
            if (name == "sgx_encl_ewb")
                ewb += count;
            else if (name == "sgx_encl_eldu")
                eldu += count;
            else if (name == "sgx_reclaim_pages")
                reclaim += count;
        }
    }
    writeTraceControl("current_tracer", "nop\n");
    writeTraceControl("set_ftrace_filter", "\n");
    if (readAny)
    {
        summary.epcPageProfileAvailable = true;
        summary.epcPageOutAttempts = ewb;
        summary.epcPageInAttempts = eldu;
        summary.epcReclaimBatches = reclaim;
    }
}

// Sample ksgxd CPU ticks around each ECALL, excluding enclave creation.
long long readKsgxdSystemTicks()
{
    FILE *pipe = popen("pgrep -x ksgxd 2>/dev/null", "r");
    if (!pipe)
        return -1;
    long pid = -1;
    const int matched = std::fscanf(pipe, "%ld", &pid);
    pclose(pipe);
    if (matched != 1 || pid <= 0)
        return -1;

    std::ifstream input("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!std::getline(input, line))
        return -1;
    const std::size_t closeParen = line.rfind(')');
    if (closeParen == std::string::npos || closeParen + 2 >= line.size())
        return -1;

    std::istringstream fields(line.substr(closeParen + 2));
    std::string value;
    // The substring starts at field 3 (state). Field 15 (stime) is token 13.
    for (int token = 0; token <= 12; ++token)
    {
        if (!(fields >> value))
            return -1;
    }
    try
    {
        return std::stoll(value);
    }
    catch (...)
    {
        return -1;
    }
}

void printEpcReclaimProxy(const RunSummary &s, const char *prefix)
{
    if (s.ksgxdSystemTicksBefore >= 0 && s.ksgxdSystemTicksAfter >= 0)
    {
        const long long delta = std::max(0LL,
            s.ksgxdSystemTicksAfter - s.ksgxdSystemTicksBefore);
        const long ticksPerSecond = sysconf(_SC_CLK_TCK);
        const double seconds = ticksPerSecond > 0
            ? static_cast<double>(delta) / static_cast<double>(ticksPerSecond)
            : 0.0;
        std::cout << prefix << "EPC reclaim proxy: ksgxd_stime_delta_ticks="
                  << delta << " ksgxd_cpu_s=" << seconds << "\n";
    }
    if (s.epcPageProfileAvailable)
        std::cout << prefix << "EPC page activity: page_out_attempts="
                  << s.epcPageOutAttempts << " page_in_attempts="
                  << s.epcPageInAttempts << " reclaim_batches="
                  << s.epcReclaimBatches << "\n";
}

long long stageMetric(const RunSummary &s, int idx)
{
    if (idx < 0 || idx >= (int)s.stageMs.size())
        return -1;
    return (long long)s.stageMs[idx];
}

double bytesToMB(long long bytes)
{
    return bytes > 0 ? static_cast<double>(bytes) / (1024.0 * 1024.0) : 0.0;
}

void printMemorySummary(const RunSummary &s, const char *prefix)
{
    if (stageMetric(s, StageMemoryAvailable) != 1)
    {
        std::cout << prefix
                  << "Memory profile is not available in this build; run the command through exec.sh --memory-profile.\n";
        return;
    }

    const long long startLive = stageMetric(s, StageMemoryStartLive);
    const long long peakLive = stageMetric(s, StageMemoryPeakLive);
    const long long endLive = stageMetric(s, StageMemoryEndLive);
    const long long peakIncrease = peakLive > startLive ? peakLive - startLive : 0;
    const long long released = peakLive > endLive ? peakLive - endLive : 0;
    const long long branchBound =
        stageMetric(s, StageMemoryBranchConcurrentUpperBound);
    const long long concurrentPeak = std::max(peakLive, branchBound);
    const long long concurrentIncrease =
        concurrentPeak > startLive ? concurrentPeak - startLive : 0;

    std::cout << prefix << modeName(s.mode) << " memory (tracked C++ heap, not EPC):\n"
              << prefix << "  live at join start: " << bytesToMB(startLive) << " MB\n"
              << prefix << "  peak live: " << bytesToMB(peakLive) << " MB\n"
              << prefix << "  peak increase during join: " << bytesToMB(peakIncrease) << " MB\n"
              << prefix << "  live after join: " << bytesToMB(endLive) << " MB\n"
              << prefix << "  released after peak: " << bytesToMB(released) << " MB\n"
              << prefix << "  total allocated during join: "
              << bytesToMB(stageMetric(s, StageMemoryTotalAllocated)) << " MB (cumulative)\n";

    if (s.mode == ModeOurs)
    {
        std::cout << prefix << "  position propagation allocated: "
                  << bytesToMB(stageMetric(s, StageMemoryPositionAllocated)) << " MB (cumulative)\n"
                  << prefix << "  position propagation peak live: "
                  << bytesToMB(stageMetric(s, StageMemoryPositionPeakLive)) << " MB\n"
                  << prefix << "  position propagation peak increase: "
                  << bytesToMB(stageMetric(s, StageMemoryPositionPeakIncrease)) << " MB\n"
                  << prefix << "  copy tables: "
                  << bytesToMB(stageMetric(s, StageMemoryCopyTables)) << " MB\n"
                  << prefix << "  final output: "
                  << bytesToMB(stageMetric(s, StageMemoryFinalOutput)) << " MB\n"
                  << prefix << "  estimated peak if sibling branches run concurrently (upper bound): "
                  << bytesToMB(concurrentPeak) << " MB\n"
                  << prefix << "  estimated concurrent peak increase (upper bound): "
                  << bytesToMB(concurrentIncrease) << " MB\n";
    }
}

void printDOPaddingSummary(const RunSummary &s, const char *prefix)
{
    long long target = stageMetric(s, 9);
    if (target < 0)
        return;
    std::cout << prefix << "Materialization target Lambda=" << target
              << " (private cardinality and sensitivity not exported)\n";
}

bool isDOPaddingStatsOnly(const RunSummary &s)
{
    constexpr long long MaxMaterializedProtectedRows = 50000000;
    long long exactRows = stageMetric(s, 7);
    long long protectedRows = stageMetric(s, 9);
    return s.rows == 0 && s.cols == 0 &&
           exactRows > 0 && protectedRows > MaxMaterializedProtectedRows;
}

bool usedExactRowsInsteadOfProtectedRows(const RunSummary &s)
{
    long long exactRows = stageMetric(s, 7);
    long long protectedRows = stageMetric(s, 9);
    return exactRows > 0 && protectedRows > exactRows &&
           s.rows == exactRows && s.rows > 0;
}

bool expectedRowsMatch(const RunSummary &s,
                       long long expectedRows,
                       bool materializePadding)
{
    if (expectedRows < 0)
        return true;
    if (s.paddingStatsOnly && s.rows == 0 && s.cols == 0)
        return stageMetric(s, 7) == expectedRows;
    if (s.paddingStatsOnly)
        return s.rows == expectedRows;
    return materializePadding ? s.rows >= expectedRows : s.rows == expectedRows;
}

void printModeBanner(const char *label, int threads)
{
    std::cout << "\n==============================\n"
              << "Mode: " << label << "  threads=" << threads << "\n"
              << "==============================\n";
}

double joinOnlyMs(const RunSummary &s)
{
    if ((int)s.stageMs.size() > 4)
        return s.stageMs[3] + s.stageMs[4];
    return s.ms;
}

double stageTime(const RunSummary &s, int idx)
{
    if (idx < 0 || idx >= (int)s.stageMs.size())
        return 0.0;
    return s.stageMs[idx];
}

double oursFullCompareMs(const RunSummary &s)
{
    return stageTime(s, StageOursUpFilter) +
           stageTime(s, StageOursRootExpand) +
           stageTime(s, StageJFYanDown);
}

double oursWithoutFirstUpMs(const RunSummary &s)
{
    return stageTime(s, StageOursRootExpand) + stageTime(s, StageJFYanDown);
}

double parYanWithoutFirstUpMs(const RunSummary &s)
{
    return stageTime(s, StageParYanDownFilter) + stageTime(s, StageParYanJoin);
}

int primitivePhaseStageIndex(int phase, int kind)
{
    return StagePrimitivePhaseBase + phase * PrimitiveKindCount + kind;
}

void printPrimitivePhaseRow(const RunSummary &s, const char *prefix, const char *label, int phase)
{
    std::cout << prefix << label
              << ": sort=" << stageTime(s, primitivePhaseStageIndex(phase, 0))
              << " ms  expand=" << stageTime(s, primitivePhaseStageIndex(phase, 1))
              << " ms  compact=" << stageTime(s, primitivePhaseStageIndex(phase, 2))
              << " ms  aggtree=" << stageTime(s, primitivePhaseStageIndex(phase, 3))
              << " ms\n";
}

void printPrimitiveCombinedPhaseRow(const RunSummary &s, const char *prefix, const char *label, int phaseA, int phaseB)
{
    std::cout << prefix << label
              << ": sort=" << stageTime(s, primitivePhaseStageIndex(phaseA, 0)) + stageTime(s, primitivePhaseStageIndex(phaseB, 0))
              << " ms  expand=" << stageTime(s, primitivePhaseStageIndex(phaseA, 1)) + stageTime(s, primitivePhaseStageIndex(phaseB, 1))
              << " ms  compact=" << stageTime(s, primitivePhaseStageIndex(phaseA, 2)) + stageTime(s, primitivePhaseStageIndex(phaseB, 2))
              << " ms  aggtree=" << stageTime(s, primitivePhaseStageIndex(phaseA, 3)) + stageTime(s, primitivePhaseStageIndex(phaseB, 3))
              << " ms\n";
}

void printPrimitiveSummaryForMode(const RunSummary &s, const char *prefix)
{
    std::cout << prefix << modeName(s.mode) << " primitives:"
              << " sort=" << stageTime(s, StagePrimitiveSort)
              << " ms  expand=" << stageTime(s, StagePrimitiveExpand)
              << " ms  compact=" << stageTime(s, StagePrimitiveCompact)
              << " ms  aggtree=" << stageTime(s, StagePrimitiveAggtree)
              << " ms\n";
}

void printPrimitivePhaseSummaryForMode(const RunSummary &s, const char *prefix)
{
    if (s.mode == ModeOurs)
    {
        printPrimitivePhaseRow(s, prefix, "JFYan.upFilter", 0);
        printPrimitiveCombinedPhaseRow(s, prefix, "JFYan.down", 1, 2);
    }
    else if (s.mode == ModeParYan)
    {
        printPrimitivePhaseRow(s, prefix, "ParYan.upFilter", 3);
        printPrimitivePhaseRow(s, prefix, "ParYan.downFilter", 4);
        printPrimitivePhaseRow(s, prefix, "ParYan.join", 5);
    }
    else if (s.mode == ModeObliYan)
    {
        printPrimitivePhaseRow(s, prefix, "ObliYan.upFilter", 6);
        printPrimitivePhaseRow(s, prefix, "ObliYan.downFilter", 7);
        printPrimitivePhaseRow(s, prefix, "ObliYan.join", 8);
    }
}

void printComparisonSummary(const std::vector<RunSummary> &summaries, const char *prefix)
{
    const RunSummary *ours = nullptr;
    const RunSummary *parYan = nullptr;
    const RunSummary *obliYan = nullptr;
    for (const RunSummary &s : summaries)
    {
        if (s.mode == ModeOurs)
            ours = &s;
        else if (s.mode == ModeParYan)
            parYan = &s;
        else if (s.mode == ModeObliYan)
            obliYan = &s;
    }
    if (!ours)
        return;

    double oursFull = oursFullCompareMs(*ours);
    double oursNoFirstUp = oursWithoutFirstUpMs(*ours);
    if (oursFull <= 0.0)
        oursFull = joinOnlyMs(*ours);

    std::cout << prefix << "JFYan: upFilter=" << stageTime(*ours, StageOursUpFilter)
              << " ms  down=" << oursNoFirstUp
              << " ms  full=" << oursFull << " ms\n";

    if (parYan)
    {
        double parYanNoFirstUp = parYanWithoutFirstUpMs(*parYan);
        std::cout << prefix << "ParYan: upFilter=" << stageTime(*parYan, StageParYanUpFilter)
                  << " ms  downFilter=" << stageTime(*parYan, StageParYanDownFilter)
                  << " ms  upJoin=" << stageTime(*parYan, StageParYanJoin)
                  << " ms  compare=" << parYanNoFirstUp << " ms";
        if (oursNoFirstUp > 0.0 && parYanNoFirstUp > 0.0)
            std::cout << "  ratio=" << (parYanNoFirstUp / oursNoFirstUp) << "x";
        std::cout << "\n";
    }

    if (obliYan)
    {
        double obliYanMs = joinOnlyMs(*obliYan);
        std::cout << prefix << "ObliYan: upFilter=" << stageTime(*obliYan, StageObliYanUpFilter)
                  << " ms  downFilter=" << stageTime(*obliYan, StageObliYanDownFilter)
                  << " ms  join=" << stageTime(*obliYan, StageObliYanJoin)
                  << " ms  full=" << obliYanMs << " ms";
        if (oursFull > 0.0 && obliYanMs > 0.0)
            std::cout << "  ratio=" << (obliYanMs / oursFull) << "x";
        std::cout << "\n";
    }

    if (obliYan || parYan)
    {
        std::cout << prefix << "Ratios:";
        if (obliYan && oursFull > 0.0)
        {
            double obliYanMs = joinOnlyMs(*obliYan);
            if (obliYanMs > 0.0)
                std::cout << " ObliYan/JFYan=" << (obliYanMs / oursFull) << "x";
        }
        if (obliYan && parYan)
        {
            double obliYanMs = joinOnlyMs(*obliYan);
            double parYanTotal = joinOnlyMs(*parYan);
            if (obliYanMs > 0.0 && parYanTotal > 0.0)
                std::cout << " ObliYan/ParYan(total)=" << (obliYanMs / parYanTotal) << "x";
        }
        if (parYan)
        {
            double parYanDownJoin = parYanWithoutFirstUpMs(*parYan);
            if (parYanDownJoin > 0.0 && oursNoFirstUp > 0.0)
                std::cout << " ParYan(down+join)/JFYan(down)="
                          << (parYanDownJoin / oursNoFirstUp) << "x";
        }
        std::cout << "\n";
    }

    if (ours || parYan || obliYan)
    {
        std::cout << prefix << "Primitive totals:\n";
        if (ours)
            printPrimitiveSummaryForMode(*ours, (std::string(prefix) + "  ").c_str());
        if (parYan)
            printPrimitiveSummaryForMode(*parYan, (std::string(prefix) + "  ").c_str());
        if (obliYan)
            printPrimitiveSummaryForMode(*obliYan, (std::string(prefix) + "  ").c_str());
        std::cout << prefix << "Primitive by stage:\n";
        if (ours)
            printPrimitivePhaseSummaryForMode(*ours, (std::string(prefix) + "  ").c_str());
        if (parYan)
            printPrimitivePhaseSummaryForMode(*parYan, (std::string(prefix) + "  ").c_str());
        if (obliYan)
            printPrimitivePhaseSummaryForMode(*obliYan, (std::string(prefix) + "  ").c_str());
    }
}

RunSummary runMode(sgx_enclave_id_t eid,
                   int threads,
                   int mode,
                   int tauOrOutSize,
                   const FlatTables &flat,
                   int tableCount,
                   const std::vector<int> &parent,
                   int root,
                   const std::vector<int> &joinColInParent,
                   const std::vector<int> &joinColInChild,
                   const std::vector<int> &tableKeys,
                   std::vector<int> &result,
                   double doEpsilon,
                   double doDelta,
                   int materializePadding,
                   bool benchmarkOnly,
                   bool profile,
                   bool stageProfile,
                   bool memoryProfile,
                   bool epcPageProfile)
{
    RunSummary summary;
    summary.mode = mode;

    if (!benchmarkOnly && !profile)
        std::fill(result.begin(), result.end(), 0);
    const bool pageProfileStarted = epcPageProfile && beginEpcPageProfile();
    summary.ksgxdSystemTicksBefore = readKsgxdSystemTicks();
    auto t0 = std::chrono::high_resolution_clock::now();
    if (profile || stageProfile)
    {
        summary.stageMs.assign(ProfileStageCount, 0.0);
        if (profile)
        {
            summary.callStatus = AcyclicJoinRunProfile(eid,
                                                       &summary.ecallStatus,
                                                       threads,
                                                       mode,
                                                       const_cast<int *>(flat.data.data()),
                                                       (int)flat.data.size(),
                                                       const_cast<int *>(flat.offsets.data()),
                                                       const_cast<int *>(flat.rows.data()),
                                                       const_cast<int *>(flat.cols.data()),
                                                       tableCount,
                                                       const_cast<int *>(parent.data()),
                                                       root,
                                                       const_cast<int *>(joinColInParent.data()),
                                                       const_cast<int *>(joinColInChild.data()),
                                                       const_cast<int *>(tableKeys.data()),
                                                       tauOrOutSize,
                                                       materializePadding,
                                                       memoryProfile ? 1 : 0,
                                                       doEpsilon,
                                                       doDelta,
                                                       &summary.rows,
                                                       &summary.cols,
                                                       summary.stageMs.data(),
                                                       (int)summary.stageMs.size());
        }
        else
        {
            summary.callStatus = AcyclicJoinRunStageSummary(eid,
                                                            &summary.ecallStatus,
                                                            threads,
                                                            mode,
                                                            const_cast<int *>(flat.data.data()),
                                                            (int)flat.data.size(),
                                                            const_cast<int *>(flat.offsets.data()),
                                                            const_cast<int *>(flat.rows.data()),
                                                            const_cast<int *>(flat.cols.data()),
                                                            tableCount,
                                                            const_cast<int *>(parent.data()),
                                                            root,
                                                            const_cast<int *>(joinColInParent.data()),
                                                            const_cast<int *>(joinColInChild.data()),
                                                            const_cast<int *>(tableKeys.data()),
                                                            tauOrOutSize,
                                                            materializePadding,
                                                            memoryProfile ? 1 : 0,
                                                            doEpsilon,
                                                            doDelta,
                                                            &summary.rows,
                                                            &summary.cols,
                                                            summary.stageMs.data(),
                                                            (int)summary.stageMs.size());
        }
    }
    else if (benchmarkOnly)
    {
        summary.callStatus = AcyclicJoinRunSummary(eid,
                                                   &summary.ecallStatus,
                                                   threads,
                                                   mode,
                                                   const_cast<int *>(flat.data.data()),
                                                   (int)flat.data.size(),
                                                   const_cast<int *>(flat.offsets.data()),
                                                   const_cast<int *>(flat.rows.data()),
                                                   const_cast<int *>(flat.cols.data()),
                                                   tableCount,
                                                   const_cast<int *>(parent.data()),
                                                   root,
                                                   const_cast<int *>(joinColInParent.data()),
                                                   const_cast<int *>(joinColInChild.data()),
                                                   const_cast<int *>(tableKeys.data()),
                                                   tauOrOutSize,
                                                   materializePadding,
                                                   memoryProfile ? 1 : 0,
                                                   doEpsilon,
                                                   doDelta,
                                                   &summary.rows,
                                                   &summary.cols);
    }
    else
    {
        summary.callStatus = AcyclicJoinRun(eid,
                                            &summary.ecallStatus,
                                            threads,
                                            mode,
                                            const_cast<int *>(flat.data.data()),
                                            (int)flat.data.size(),
                                            const_cast<int *>(flat.offsets.data()),
                                            const_cast<int *>(flat.rows.data()),
                                            const_cast<int *>(flat.cols.data()),
                                            tableCount,
                                            const_cast<int *>(parent.data()),
                                            root,
                                            const_cast<int *>(joinColInParent.data()),
                                            const_cast<int *>(joinColInChild.data()),
                                            const_cast<int *>(tableKeys.data()),
                                            tauOrOutSize,
                                            materializePadding,
                                            memoryProfile ? 1 : 0,
                                            doEpsilon,
                                            doDelta,
                                            result.data(),
                                            (int)result.size(),
                                            &summary.rows,
                                            &summary.cols);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    summary.ksgxdSystemTicksAfter = readKsgxdSystemTicks();
    if (pageProfileStarted)
        endEpcPageProfile(summary);
    summary.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return summary;
}

sgx_enclave_id_t createEnclave()
{
    sgx_enclave_id_t eid = 0;
    const char *signedSo = "build/enclave.signed.so";
    sgx_status_t ret = sgx_create_enclave(signedSo, SGX_DEBUG_FLAG, nullptr, nullptr, &eid, nullptr);
    if (ret != SGX_SUCCESS)
    {
        signedSo = "enclave.signed.so";
        ret = sgx_create_enclave(signedSo, SGX_DEBUG_FLAG, nullptr, nullptr, &eid, nullptr);
    }
    if (ret != SGX_SUCCESS)
    {
        std::fprintf(stderr, "sgx_create_enclave failed: %#x\n", ret);
        die("Make sure enclave.signed.so is in the project root or build/enclave.signed.so exists.");
    }
    return eid;
}
}

void ocall_print(const char *s)
{
    std::printf("%s\n", s);
}

void ocall_now_ms(double *out_ms)
{
    if (!out_ms)
        return;
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    *out_ms = std::chrono::duration<double, std::milli>(now).count();
}

int main(int argc, char **argv)
{
    int mode = ModeOurs;
    bool runAll = false;
    bool benchmarkOnly = false;
    bool profile = false;
    bool stageProfile = false;
    bool memoryProfile = false;
    bool epcPageProfile = false;
    bool warmupJFYan = false;
    int maxCells = 1000000;
    int tauOrOutSize = 0;
    int printLimit = 20;
    std::string sql18Dir;
    std::string sql85Dir;
    std::string sql85Chain3Dir;
    std::string sql85ReturnsStarDir;
    std::string tpch9Dir;
    std::string job1aDir;
    std::string balancedTreeDir;
    double doEpsilon = 1.0;
    double doDelta = 1e-9;
    int materializePadding = ExactSizeBenchmark;
    bool simulationRequested = false;
    bool simulationParametersExplicit = false;
    bool noPaddingRequested = false;
    bool materializePaddingExplicit = false;
    std::vector<int> threadSweep;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-JFYan") == 0)
            mode = ModeOurs;
        else if (std::strcmp(argv[i], "-ParYan") == 0)
            mode = ModeParYan;
        else if (std::strcmp(argv[i], "-ObliYan") == 0)
            mode = ModeObliYan;
        else if (std::strcmp(argv[i], "-NonObliJFYan") == 0)
            mode = ModeNonObliviousJFYan;
        else if (std::strcmp(argv[i], "--all") == 0)
            runAll = true;
        else if (std::strcmp(argv[i], "--bench-only") == 0 || std::strcmp(argv[i], "--no-result") == 0)
            benchmarkOnly = true;
        else if (std::strcmp(argv[i], "--profile") == 0)
        {
            profile = true;
            benchmarkOnly = true;
        }
        else if (std::strcmp(argv[i], "--stage-profile") == 0)
        {
            stageProfile = true;
            benchmarkOnly = true;
        }
        else if (std::strcmp(argv[i], "--memory-profile") == 0)
        {
            memoryProfile = true;
            stageProfile = true;
            benchmarkOnly = true;
        }
        else if (std::strcmp(argv[i], "--epc-page-profile") == 0)
            epcPageProfile = true;
        else if (std::strcmp(argv[i], "--warmup-jfyan") == 0)
            warmupJFYan = true;
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            globalThreadCount = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--thread-sweep") == 0 && i + 1 < argc)
            threadSweep = parseThreadList(argv[++i]);
        else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc)
            maxCells = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-tau") == 0 && i + 1 < argc)
            tauOrOutSize = parsePositiveTarget(argv[++i]);
        else if (std::strcmp(argv[i], "--public-target") == 0 && i + 1 < argc)
        {
            tauOrOutSize = parsePositiveTarget(argv[++i]);
            materializePadding = PublicTarget;
            materializePaddingExplicit = true;
        }
        else if (std::strcmp(argv[i], "--benchmark-target-simulation") == 0)
        {
            simulationRequested = true;
            materializePaddingExplicit = true;
        }
        else if (std::strcmp(argv[i], "--do-epsilon") == 0 && i + 1 < argc)
        {
            doEpsilon = std::atof(argv[++i]);
            simulationParametersExplicit = true;
        }
        else if (std::strcmp(argv[i], "--do-delta") == 0 && i + 1 < argc)
        {
            doDelta = std::atof(argv[++i]);
            simulationParametersExplicit = true;
        }
        else if (std::strcmp(argv[i], "--materialize-padding") == 0)
        {
            materializePadding = PublicTarget;
            materializePaddingExplicit = true;
        }
        else if (std::strcmp(argv[i], "--no-materialize-padding") == 0)
        {
            materializePadding = ExactSizeBenchmark;
            noPaddingRequested = true;
            materializePaddingExplicit = true;
        }
        else if (std::strcmp(argv[i], "--sql18") == 0 && i + 1 < argc)
            sql18Dir = argv[++i];
        else if (std::strcmp(argv[i], "--sql85") == 0 && i + 1 < argc)
            sql85Dir = argv[++i];
        else if (std::strcmp(argv[i], "--sql85-chain3") == 0 && i + 1 < argc)
            sql85Chain3Dir = argv[++i];
        else if (std::strcmp(argv[i], "--sql85-returns-star") == 0 && i + 1 < argc)
            sql85ReturnsStarDir = argv[++i];
        else if (std::strcmp(argv[i], "--tpch9") == 0 && i + 1 < argc)
            tpch9Dir = argv[++i];
        else if (std::strcmp(argv[i], "--job1a") == 0 && i + 1 < argc)
            job1aDir = argv[++i];
        else if ((std::strcmp(argv[i], "--tree-workload") == 0 ||
                  std::strcmp(argv[i], "--balanced-tree") == 0) &&
                 i + 1 < argc)
            balancedTreeDir = argv[++i];
        else if (std::strcmp(argv[i], "--print-limit") == 0 && i + 1 < argc)
            printLimit = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (globalThreadCount <= 0 || maxCells <= 0)
    {
        std::cerr << "threads and max_cells must be positive.\n";
        return 1;
    }
    for (int t : threadSweep)
    {
        if (t <= 0)
        {
            std::cerr << "thread sweep values must be positive.\n";
            return 1;
        }
    }
    if (!std::isfinite(doEpsilon) || !std::isfinite(doDelta) ||
        doEpsilon <= 0.0 || doDelta <= 0.0 || doDelta >= 1.0)
    {
        std::cerr << "--do-epsilon must be > 0 and --do-delta must be in (0, 1).\n";
        return 1;
    }

    int datasetCount = (!sql18Dir.empty() ? 1 : 0) +
                       (!sql85Dir.empty() ? 1 : 0) +
                       (!sql85Chain3Dir.empty() ? 1 : 0) +
                       (!sql85ReturnsStarDir.empty() ? 1 : 0) +
                       (!tpch9Dir.empty() ? 1 : 0) +
                       (!job1aDir.empty() ? 1 : 0) +
                       (!balancedTreeDir.empty() ? 1 : 0);
    if (datasetCount > 1)
    {
        std::cerr << "Use only one dataset option: --sql18, --sql85, --sql85-chain3, --sql85-returns-star, --tpch9, --job1a, or --tree-workload.\n";
        return 1;
    }
    if (!tpch9Dir.empty() && !materializePaddingExplicit)
        materializePadding = PublicTarget;

    if (simulationParametersExplicit && !simulationRequested)
    {
        std::cerr << "--do-epsilon/--do-delta require --benchmark-target-simulation; no DP target generator is implemented.\n";
        return 1;
    }

    if (simulationRequested)
    {
        if (tauOrOutSize != 0 || noPaddingRequested)
        {
            std::cerr << "Target simulation cannot be combined with a public target or --no-materialize-padding.\n";
            return 1;
        }
        materializePadding = BenchmarkTargetSimulation;
        std::cout << "BENCHMARK TARGET-SIZE SIMULATION: deterministic sizing; no differential-privacy guarantee.\n";
    }
    if (materializePadding == PublicTarget)
    {
        if (tauOrOutSize <= 0 || (!runAll && mode == ModeNonObliviousJFYan))
        {
            std::cerr << "Public-target execution requires positive --public-target Lambda and an oblivious join mode.\n";
            return 1;
        }
        // Public-target calls expose only public dimensions; disable copy-out,
        // exact-size checks, detailed logs, and heap telemetry.
        benchmarkOnly = true;
        printLimit = 0;
        memoryProfile = false;
        if (profile)
        {
            profile = false;
            stageProfile = true;
        }
        std::cout << "Externally released public target Lambda=" << tauOrOutSize
                  << "; DP target generation is outside this harness.\n";
    }

    std::vector<int> modes;
    if (runAll)
        modes = {ModeOurs, ModeParYan, ModeObliYan};
    else
        modes = {mode};

    if (memoryProfile)
        std::cout << "Memory profiling is enabled. Reported times include profiler overhead; use a normal build for timing results.\n";

    std::vector<Table> tables;
    std::vector<int> parent;
    std::vector<int> tableKeys;
    std::vector<int> joinColInParent;
    std::vector<int> joinColInChild;
    int root = 0;
    long long expectedRows = -1;

    if (!sql18Dir.empty())
    {
        DatasetConfig cfg = loadSQL18Dataset(sql18Dir);
        tables = std::move(cfg.tables);
        parent = std::move(cfg.parent);
        tableKeys = std::move(cfg.tableKeys);
        joinColInParent = std::move(cfg.joinColInParent);
        joinColInChild = std::move(cfg.joinColInChild);
        root = cfg.root;
        expectedRows = cfg.expectedRows;
    }
    else if (!sql85Dir.empty())
    {
        DatasetConfig cfg = loadSQL85Dataset(sql85Dir);
        tables = std::move(cfg.tables);
        parent = std::move(cfg.parent);
        tableKeys = std::move(cfg.tableKeys);
        joinColInParent = std::move(cfg.joinColInParent);
        joinColInChild = std::move(cfg.joinColInChild);
        root = cfg.root;
        expectedRows = cfg.expectedRows;
    }
    else if (!sql85Chain3Dir.empty())
    {
        DatasetConfig cfg = loadSQL85Chain3Dataset(sql85Chain3Dir);
        tables = std::move(cfg.tables);
        parent = std::move(cfg.parent);
        tableKeys = std::move(cfg.tableKeys);
        joinColInParent = std::move(cfg.joinColInParent);
        joinColInChild = std::move(cfg.joinColInChild);
        root = cfg.root;
        expectedRows = cfg.expectedRows;
    }
    else if (!sql85ReturnsStarDir.empty())
    {
        DatasetConfig cfg = loadSQL85ReturnsStarDataset(sql85ReturnsStarDir);
        tables = std::move(cfg.tables);
        parent = std::move(cfg.parent);
        tableKeys = std::move(cfg.tableKeys);
        joinColInParent = std::move(cfg.joinColInParent);
        joinColInChild = std::move(cfg.joinColInChild);
        root = cfg.root;
        expectedRows = cfg.expectedRows;
    }
    else if (!tpch9Dir.empty())
    {
        DatasetConfig cfg = loadTPCH9Dataset(tpch9Dir);
        tables = std::move(cfg.tables);
        parent = std::move(cfg.parent);
        tableKeys = std::move(cfg.tableKeys);
        joinColInParent = std::move(cfg.joinColInParent);
        joinColInChild = std::move(cfg.joinColInChild);
        root = cfg.root;
        expectedRows = cfg.expectedRows;
    }
    else if (!job1aDir.empty())
    {
        DatasetConfig cfg = loadJOB1ADataset(job1aDir);
        tables = std::move(cfg.tables);
        parent = std::move(cfg.parent);
        tableKeys = std::move(cfg.tableKeys);
        joinColInParent = std::move(cfg.joinColInParent);
        joinColInChild = std::move(cfg.joinColInChild);
        root = cfg.root;
        expectedRows = cfg.expectedRows;
    }
    else if (!balancedTreeDir.empty())
    {
        DatasetConfig cfg = loadTreeWorkloadDataset(balancedTreeDir);
        tables = std::move(cfg.tables);
        parent = std::move(cfg.parent);
        tableKeys = std::move(cfg.tableKeys);
        joinColInParent = std::move(cfg.joinColInParent);
        joinColInChild = std::move(cfg.joinColInChild);
        root = cfg.root;
        expectedRows = cfg.expectedRows;
        std::cout << "Tree workload: relations=" << tables.size() << "\n";
    }
    else
    {
        tables = sampleTables();
        parent = {-1, 0, 0, 0, 3, 3};
        tableKeys = {3, 2, 2, 2, 2, 2};
        joinColInParent = {-1, 0, 1, 2, 0, 1};
        joinColInChild = {-1, 1, 0, 0, 0, 0};
    }

    if (materializePadding == PublicTarget || tauOrOutSize > 0)
        expectedRows = -1; // Padded rows do not validate the exact result size.
    const int tableCount = static_cast<int>(tables.size());
    FlatTables flat = flattenTables(tables);
    // Release duplicate row storage before enclave creation.
    vector<Table>().swap(tables);
    malloc_trim(0);
    std::vector<int> result(maxCells, 0);
    sgx_enclave_id_t eid = createEnclave();

    if (warmupJFYan)
    {
        const int warmupThreads = threadSweep.empty() ? globalThreadCount : threadSweep.front();
        std::cout << "\n=== unreported JFYan warm-up: threads=" << warmupThreads << " ===\n";
        RunSummary warmup = runMode(eid, warmupThreads, ModeOurs, tauOrOutSize,
                                    flat, tableCount, parent, root,
                                    joinColInParent, joinColInChild, tableKeys,
                                    result, doEpsilon, doDelta, materializePadding,
                                    true, profile, stageProfile, memoryProfile,
                                    epcPageProfile);
        if (warmup.callStatus != SGX_SUCCESS || warmup.ecallStatus != SGX_SUCCESS ||
            !expectedRowsMatch(warmup, expectedRows, materializePadding))
        {
            std::fprintf(stderr,
                         "JFYan warm-up failed: call=%#x (%s) ecall=%#x (%s)\n",
                         warmup.callStatus, sgxStatusName(warmup.callStatus),
                         warmup.ecallStatus, sgxStatusName(warmup.ecallStatus));
            sgx_destroy_enclave(eid);
            return 1;
        }
        std::cout << "Warm-up completed and passed the output check; it is not part of the reported measurements.\n";
    }

    if (!threadSweep.empty())
    {
        std::cout << "Thread sweep:";
        for (int t : threadSweep)
            std::cout << " " << t;
        std::cout << "\n";

        for (int threads : threadSweep)
        {
            std::cout << "\n=== threads=" << threads << " ===\n";
            std::vector<RunSummary> sweepSummaries;
            sweepSummaries.reserve(modes.size());
            int runTauOrOutSize = tauOrOutSize;
            for (int m : modes)
            {
                printModeBanner(modeName(m), threads);
                RunSummary s;
                s = runMode(eid, threads, m, runTauOrOutSize, flat, tableCount,
                            parent, root, joinColInParent, joinColInChild,
                            tableKeys, result, doEpsilon, doDelta, materializePadding,
                            true, profile, stageProfile, memoryProfile,
                            epcPageProfile);
                s.paddingStatsOnly = isDOPaddingStatsOnly(s);
                sweepSummaries.push_back(s);
                if (s.callStatus != SGX_SUCCESS || s.ecallStatus != SGX_SUCCESS)
                {
                    std::fprintf(stderr, "%s failed: call=%#x (%s) ecall=%#x (%s)\n",
                                 modeName(m),
                                 s.callStatus, sgxStatusName(s.callStatus),
                                 s.ecallStatus, sgxStatusName(s.ecallStatus));
                    sgx_destroy_enclave(eid);
                    return 1;
                }
                std::cout << "  " << modeName(m) << ": ";
                if (s.skipped)
                    std::cout << "skipped";
                else
                    std::cout << s.ms << " ms";
                if ((profile || stageProfile) && !s.stageMs.empty() &&
                    (!s.paddingStatsOnly || s.rows > 0 || s.cols > 0))
                    std::cout << "  join-only=" << joinOnlyMs(s) << " ms";
                if (s.paddingStatsOnly && s.rows == 0 && s.cols == 0)
                    std::cout << "  stats-only";
                else if (s.paddingStatsOnly)
                    std::cout << "  protected-output-not-materialized rows=" << s.rows << " cols=" << s.cols;
                else
                    std::cout << "  rows=" << s.rows << " cols=" << s.cols;
                if (expectedRows >= 0)
                {
                    const bool checkOk = expectedRowsMatch(s, expectedRows, materializePadding);
                    if (s.paddingStatsOnly && s.rows == 0 && s.cols == 0)
                        std::cout << "  stats-check=" << (checkOk ? "OK" : "FAILED");
                    else
                        std::cout << "  check=" << (checkOk ? "OK" : "FAILED");
                }
                std::cout << "\n";
                printEpcReclaimProxy(s, "    ");
                if ((profile || stageProfile) && !s.stageMs.empty())
                {
                    std::cout << "    stages:";
                    for (int i = 0; i < (int)s.stageMs.size(); ++i)
                    {
                        if (stageIsMemoryMetric(i) || (stageIsMetric(i) && s.stageMs[i] < 0))
                            continue;
                        std::cout << " " << stageName(m, i) << "=";
                        if (stageIsMetric(i))
                            std::cout << (long long)s.stageMs[i];
                        else
                            std::cout << s.stageMs[i];
                    }
                    std::cout << "\n";
                    printDOPaddingSummary(s, "    ");
                    if (memoryProfile)
                        printMemorySummary(s, "    ");
                }
                if (!expectedRowsMatch(s, expectedRows, materializePadding))
                {
                    std::cerr << "Expected-row validation failed for " << modeName(m) << ".\n";
                    sgx_destroy_enclave(eid);
                    return 1;
                }
                if (!materializePadding && expectedRows < 0 && runTauOrOutSize <= 0 && s.mode == ModeOurs)
                    runTauOrOutSize = s.rows;
            }

            if (runAll)
            {
                if (sweepSummaries.front().paddingStatsOnly &&
                    sweepSummaries.front().rows == 0 && sweepSummaries.front().cols == 0)
                {
                    std::cout << "  ratios: not measured (protected output stats only)\n";
                }
                else
                {
                    if (usedExactRowsInsteadOfProtectedRows(sweepSummaries.front()))
                        std::cout << "  DO protected output was too large to materialize; ratios use exact real output profiling.\n";
                    printComparisonSummary(sweepSummaries, "  ");
                }
            }
        }

        sgx_destroy_enclave(eid);
        return 0;
    }

    std::vector<RunSummary> summaries;
    summaries.reserve(modes.size());
    bool resultCheckFailed = false;
    int runTauOrOutSize = tauOrOutSize;
    for (int m : modes)
    {
        printModeBanner(modeName(m), globalThreadCount);
        RunSummary s;
        s = runMode(eid, globalThreadCount, m, runTauOrOutSize, flat, tableCount,
                    parent, root, joinColInParent, joinColInChild,
                    tableKeys, result, doEpsilon, doDelta, materializePadding,
                    benchmarkOnly, profile, stageProfile, memoryProfile,
                    epcPageProfile);
        s.paddingStatsOnly = isDOPaddingStatsOnly(s);
        summaries.push_back(s);
        if (s.callStatus != SGX_SUCCESS || s.ecallStatus != SGX_SUCCESS)
        {
            std::fprintf(stderr, "%s failed: call=%#x (%s) ecall=%#x (%s)\n",
                         modeName(m),
                         s.callStatus, sgxStatusName(s.callStatus),
                         s.ecallStatus, sgxStatusName(s.ecallStatus));
            sgx_destroy_enclave(eid);
            return 1;
        }

        std::cout << "Mode: " << modeName(m) << "\n";
        if (s.paddingStatsOnly && s.rows == 0 && s.cols == 0)
            std::cout << "Result: not materialized (DO padding stats only)\n";
        else if (s.paddingStatsOnly)
            std::cout << "Result: " << s.rows << " rows x " << s.cols
                      << " cols (DO protected output not materialized)\n";
        else
            std::cout << "Result: " << s.rows << " rows x " << s.cols << " cols\n";
        if (s.skipped)
            std::cout << "Profile ECALL time: skipped (same DO padding stats as first mode)\n";
        else
        {
            std::cout << (profile ? "Profile ECALL time: " : (stageProfile ? "Stage-profile ECALL time: " : (benchmarkOnly ? "Benchmark ECALL time: " : "ECALL time: "))) << s.ms << " ms\n";
            printEpcReclaimProxy(s, "");
            if ((profile || stageProfile) && !s.stageMs.empty())
                std::cout << "Join-only time: " << joinOnlyMs(s) << " ms\n";
        }
        if (expectedRows >= 0)
        {
            const bool checkOk = expectedRowsMatch(s, expectedRows, materializePadding);
            if (s.paddingStatsOnly && s.rows == 0 && s.cols == 0)
                std::cout << "Expected real rows: " << expectedRows << "  stats check="
                          << (checkOk ? "OK" : "FAILED") << "\n";
            else if (s.paddingStatsOnly)
                std::cout << "Expected real rows: " << expectedRows << "  join check="
                          << (checkOk ? "OK" : "FAILED") << "\n";
            else
                std::cout << "Expected real rows: " << expectedRows
                          << (materializePadding ? "  protected check=" : "  exact check=")
                          << (checkOk ? "OK" : "FAILED")
                          << "\n";
            resultCheckFailed = resultCheckFailed || !checkOk;
        }
        if ((profile || stageProfile) && !s.stageMs.empty())
        {
            std::cout << "Stage timings:\n";
            for (int i = 0; i < (int)s.stageMs.size(); ++i)
            {
                if (stageIsMemoryMetric(i) || (stageIsMetric(i) && s.stageMs[i] < 0))
                    continue;
                std::cout << "  " << stageName(m, i) << "=";
                if (stageIsMetric(i))
                    std::cout << (long long)s.stageMs[i] << "\n";
                else
                    std::cout << s.stageMs[i] << " ms\n";
            }
            printDOPaddingSummary(s, "");
            if (memoryProfile)
                printMemorySummary(s, "");
        }
        if (!materializePadding && expectedRows < 0 && runTauOrOutSize <= 0 && s.mode == ModeOurs)
            runTauOrOutSize = s.rows;
    }

    if (runAll && !summaries.empty())
    {
        std::cout << "\nSummary:\n";
        if (summaries.front().paddingStatsOnly &&
            summaries.front().rows == 0 && summaries.front().cols == 0)
        {
            std::cout << "  DO padding stats only; protected output was too large to materialize.\n";
            std::cout << "  Algorithm runtimes and ratios were not measured.\n";
        }
        else
        {
            if (summaries.front().paddingStatsOnly)
                std::cout << "  DO protected output was too large to materialize; ratios use join-only unpadded profiling.\n";
            else if (usedExactRowsInsteadOfProtectedRows(summaries.front()))
                std::cout << "  DO protected output was too large to materialize; ratios use exact real output profiling.\n";
            printComparisonSummary(summaries, "  ");
        }
    }

    const RunSummary &last = summaries.back();

    if (benchmarkOnly || profile || stageProfile || printLimit <= 0)
    {
        sgx_destroy_enclave(eid);
        return resultCheckFailed ? 1 : 0;
    }

    Table output = arrayToTable(result.data(), last.rows, last.cols);
    int rowsToPrint = std::min(printLimit, last.rows);
    for (int r = 0; r < rowsToPrint; ++r)
    {
        const auto &row = output[r];
        std::cout << "  [";
        for (int i = 0; i < (int)row.size(); ++i)
        {
            if (i)
                std::cout << ", ";
            std::cout << row[i];
        }
        std::cout << "]\n";
    }
    if (last.rows > rowsToPrint)
        std::cout << "  ... (" << (last.rows - rowsToPrint) << " more rows)\n";

    sgx_destroy_enclave(eid);
    return resultCheckFailed ? 1 : 0;
}
