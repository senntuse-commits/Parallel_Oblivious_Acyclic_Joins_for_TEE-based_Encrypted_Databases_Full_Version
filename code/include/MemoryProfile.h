#pragma once

#include <cstddef>

struct MemoryProfileSnapshot
{
    unsigned long long startLiveBytes = 0;
    unsigned long long peakLiveBytes = 0;
    unsigned long long endLiveBytes = 0;
    unsigned long long totalAllocatedBytes = 0;
    unsigned long long positionAllocatedBytes = 0;
    unsigned long long positionPeakLiveBytes = 0;
    unsigned long long positionPeakIncreaseBytes = 0;
    unsigned long long copyTableBytes = 0;
    unsigned long long finalOutputBytes = 0;
    unsigned long long branchConcurrentUpperBoundBytes = 0;
    bool available = false;
};

// Allocation tracking is enabled in --memory-profile builds; otherwise returns zeros.
bool memoryProfileAvailable();
void memoryProfileBeginRun(bool enabled);
MemoryProfileSnapshot memoryProfileEndRun();

void memoryProfileBeginPositionPropagation();
void memoryProfileEndPositionPropagation();
void memoryProfileSetCopyTableBytes(unsigned long long bytes);
void memoryProfileSetFinalOutputBytes(unsigned long long bytes);

// For sequential siblings, sum incremental peaks to bound concurrent branch memory.
// These hooks do not change the schedule.
void memoryProfileBeginBranchGroup(bool branchesAreSequential);
void memoryProfileBeginBranch();
void memoryProfileEndBranch();
void memoryProfileEndBranchGroup();

// Count allocated vector capacity, including unused cells.
template <typename TableLike>
unsigned long long memoryProfileTableBytes(const TableLike &table)
{
    unsigned long long bytes =
        static_cast<unsigned long long>(table.capacity()) * sizeof(typename TableLike::value_type);
    for (const auto &row : table)
        bytes += static_cast<unsigned long long>(row.capacity()) * sizeof(typename TableLike::value_type::value_type);
    return bytes;
}
