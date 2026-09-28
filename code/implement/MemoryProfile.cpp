#include "../include/MemoryProfile.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace
{
using Counter = unsigned long long;

std::atomic<Counter> g_currentBytes{0};
std::atomic<Counter> g_totalRequestedBytes{0};

std::atomic<bool> g_runEnabled{false};
std::atomic<Counter> g_runStartLive{0};
std::atomic<Counter> g_runStartTotal{0};
std::atomic<Counter> g_runPeakLive{0};

std::atomic<bool> g_positionEnabled{false};
std::atomic<Counter> g_positionStartLive{0};
std::atomic<Counter> g_positionStartTotal{0};
std::atomic<Counter> g_positionPeakLive{0};
std::atomic<Counter> g_positionAllocated{0};
std::atomic<Counter> g_positionPeakIncrease{0};

std::atomic<Counter> g_copyTableBytes{0};
std::atomic<Counter> g_finalOutputBytes{0};

std::atomic<bool> g_branchGroupEnabled{false};
std::atomic<bool> g_branchEnabled{false};
std::atomic<Counter> g_branchGroupStartLive{0};
std::atomic<Counter> g_branchGroupSumPeakIncreases{0};
std::atomic<Counter> g_branchStartLive{0};
std::atomic<Counter> g_branchPeakLive{0};
std::atomic<Counter> g_branchConcurrentUpperBound{0};

void updateMaximum(std::atomic<Counter> &target, Counter value)
{
    Counter old = target.load(std::memory_order_relaxed);
    while (old < value &&
           !target.compare_exchange_weak(old, value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed))
    {
    }
}

void recordAllocation(std::size_t size)
{
    const Counter bytes = static_cast<Counter>(size);
    const Counter live = g_currentBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    g_totalRequestedBytes.fetch_add(bytes, std::memory_order_relaxed);

    if (g_runEnabled.load(std::memory_order_relaxed))
        updateMaximum(g_runPeakLive, live);
    if (g_positionEnabled.load(std::memory_order_relaxed))
        updateMaximum(g_positionPeakLive, live);
    if (g_branchEnabled.load(std::memory_order_relaxed))
        updateMaximum(g_branchPeakLive, live);
}

void recordDeallocation(std::size_t size)
{
    g_currentBytes.fetch_sub(static_cast<Counter>(size), std::memory_order_relaxed);
}

#if defined(SGX_ENCLAVE_BUILD) && defined(ENABLE_MEMORY_PROFILE_ALLOCATOR)
struct alignas(std::max_align_t) AllocationHeader
{
    std::size_t requestedBytes;
};

static_assert(sizeof(AllocationHeader) % alignof(std::max_align_t) == 0,
              "Allocation header must preserve malloc alignment");

void *allocateTracked(std::size_t requested)
{
    if (requested == 0)
        requested = 1;
    if (requested > static_cast<std::size_t>(-1) - sizeof(AllocationHeader))
        throw std::bad_alloc();

    void *base = std::malloc(sizeof(AllocationHeader) + requested);
    if (!base)
        throw std::bad_alloc();

    auto *header = static_cast<AllocationHeader *>(base);
    header->requestedBytes = requested;
    recordAllocation(requested);
    return header + 1;
}

void freeTracked(void *pointer) noexcept
{
    if (!pointer)
        return;
    auto *header = static_cast<AllocationHeader *>(pointer) - 1;
    recordDeallocation(header->requestedBytes);
    std::free(header);
}
#endif
} // namespace

bool memoryProfileAvailable()
{
#if defined(SGX_ENCLAVE_BUILD) && defined(ENABLE_MEMORY_PROFILE_ALLOCATOR)
    return true;
#else
    return false;
#endif
}

void memoryProfileBeginRun(bool enabled)
{
    const bool active = enabled && memoryProfileAvailable();
    g_positionEnabled.store(false, std::memory_order_relaxed);
    g_copyTableBytes.store(0, std::memory_order_relaxed);
    g_finalOutputBytes.store(0, std::memory_order_relaxed);
    g_positionAllocated.store(0, std::memory_order_relaxed);
    g_positionPeakIncrease.store(0, std::memory_order_relaxed);
    g_positionPeakLive.store(0, std::memory_order_relaxed);
    g_branchGroupEnabled.store(false, std::memory_order_relaxed);
    g_branchEnabled.store(false, std::memory_order_relaxed);
    g_branchConcurrentUpperBound.store(0, std::memory_order_relaxed);

    const Counter live = g_currentBytes.load(std::memory_order_relaxed);
    g_runStartLive.store(live, std::memory_order_relaxed);
    g_runStartTotal.store(g_totalRequestedBytes.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
    g_runPeakLive.store(live, std::memory_order_relaxed);
    g_runEnabled.store(active, std::memory_order_release);
}

MemoryProfileSnapshot memoryProfileEndRun()
{
    MemoryProfileSnapshot snapshot;
    snapshot.available = memoryProfileAvailable() &&
                         g_runEnabled.load(std::memory_order_acquire);
    if (snapshot.available)
    {
        snapshot.startLiveBytes = g_runStartLive.load(std::memory_order_relaxed);
        snapshot.peakLiveBytes = g_runPeakLive.load(std::memory_order_relaxed);
        snapshot.endLiveBytes = g_currentBytes.load(std::memory_order_relaxed);
        snapshot.totalAllocatedBytes =
            g_totalRequestedBytes.load(std::memory_order_relaxed) -
            g_runStartTotal.load(std::memory_order_relaxed);
        snapshot.positionAllocatedBytes = g_positionAllocated.load(std::memory_order_relaxed);
        snapshot.positionPeakLiveBytes = g_positionPeakLive.load(std::memory_order_relaxed);
        snapshot.positionPeakIncreaseBytes = g_positionPeakIncrease.load(std::memory_order_relaxed);
        snapshot.copyTableBytes = g_copyTableBytes.load(std::memory_order_relaxed);
        snapshot.finalOutputBytes = g_finalOutputBytes.load(std::memory_order_relaxed);
        snapshot.branchConcurrentUpperBoundBytes =
            g_branchConcurrentUpperBound.load(std::memory_order_relaxed);
    }
    g_positionEnabled.store(false, std::memory_order_relaxed);
    g_branchEnabled.store(false, std::memory_order_relaxed);
    g_branchGroupEnabled.store(false, std::memory_order_relaxed);
    g_runEnabled.store(false, std::memory_order_release);
    return snapshot;
}

void memoryProfileBeginPositionPropagation()
{
    if (!g_runEnabled.load(std::memory_order_acquire))
        return;
    const Counter live = g_currentBytes.load(std::memory_order_relaxed);
    g_positionStartLive.store(live, std::memory_order_relaxed);
    g_positionStartTotal.store(g_totalRequestedBytes.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    g_positionPeakLive.store(live, std::memory_order_relaxed);
    g_positionEnabled.store(true, std::memory_order_release);
}

void memoryProfileEndPositionPropagation()
{
    if (!g_positionEnabled.exchange(false, std::memory_order_acq_rel))
        return;
    const Counter startLive = g_positionStartLive.load(std::memory_order_relaxed);
    const Counter peakLive = g_positionPeakLive.load(std::memory_order_relaxed);
    const Counter startTotal = g_positionStartTotal.load(std::memory_order_relaxed);
    const Counter endTotal = g_totalRequestedBytes.load(std::memory_order_relaxed);
    g_positionAllocated.store(endTotal - startTotal, std::memory_order_relaxed);
    g_positionPeakIncrease.store(peakLive > startLive ? peakLive - startLive : 0,
                                 std::memory_order_relaxed);
}

void memoryProfileSetCopyTableBytes(unsigned long long bytes)
{
    g_copyTableBytes.store(bytes, std::memory_order_relaxed);
}

void memoryProfileSetFinalOutputBytes(unsigned long long bytes)
{
    g_finalOutputBytes.store(bytes, std::memory_order_relaxed);
}

void memoryProfileBeginBranchGroup(bool branchesAreSequential)
{
    const bool active = branchesAreSequential &&
                        g_runEnabled.load(std::memory_order_acquire);
    g_branchEnabled.store(false, std::memory_order_relaxed);
    g_branchGroupEnabled.store(active, std::memory_order_release);
    if (!active)
        return;
    g_branchGroupStartLive.store(g_currentBytes.load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
    g_branchGroupSumPeakIncreases.store(0, std::memory_order_relaxed);
}

void memoryProfileBeginBranch()
{
    if (!g_branchGroupEnabled.load(std::memory_order_acquire))
        return;
    const Counter live = g_currentBytes.load(std::memory_order_relaxed);
    g_branchStartLive.store(live, std::memory_order_relaxed);
    g_branchPeakLive.store(live, std::memory_order_relaxed);
    g_branchEnabled.store(true, std::memory_order_release);
}

void memoryProfileEndBranch()
{
    if (!g_branchEnabled.exchange(false, std::memory_order_acq_rel))
        return;
    const Counter start = g_branchStartLive.load(std::memory_order_relaxed);
    const Counter peak = g_branchPeakLive.load(std::memory_order_relaxed);
    if (peak > start)
        g_branchGroupSumPeakIncreases.fetch_add(peak - start,
                                                std::memory_order_relaxed);
}

void memoryProfileEndBranchGroup()
{
    g_branchEnabled.store(false, std::memory_order_relaxed);
    if (!g_branchGroupEnabled.exchange(false, std::memory_order_acq_rel))
        return;
    const Counter estimate =
        g_branchGroupStartLive.load(std::memory_order_relaxed) +
        g_branchGroupSumPeakIncreases.load(std::memory_order_relaxed);
    updateMaximum(g_branchConcurrentUpperBound, estimate);
}

#if defined(SGX_ENCLAVE_BUILD) && defined(ENABLE_MEMORY_PROFILE_ALLOCATOR)
void *operator new(std::size_t size)
{
    return allocateTracked(size);
}

void *operator new[](std::size_t size)
{
    return allocateTracked(size);
}

void operator delete(void *pointer)
{
    freeTracked(pointer);
}

void operator delete[](void *pointer)
{
    freeTracked(pointer);
}

void operator delete(void *pointer, std::size_t)
{
    freeTracked(pointer);
}

void operator delete[](void *pointer, std::size_t)
{
    freeTracked(pointer);
}

void *operator new(std::size_t size, const std::nothrow_t &)
{
    try
    {
        return allocateTracked(size);
    }
    catch (...)
    {
        return nullptr;
    }
}

void *operator new[](std::size_t size, const std::nothrow_t &)
{
    try
    {
        return allocateTracked(size);
    }
    catch (...)
    {
        return nullptr;
    }
}

void operator delete(void *pointer, const std::nothrow_t &)
{
    freeTracked(pointer);
}

void operator delete[](void *pointer, const std::nothrow_t &)
{
    freeTracked(pointer);
}
#endif
