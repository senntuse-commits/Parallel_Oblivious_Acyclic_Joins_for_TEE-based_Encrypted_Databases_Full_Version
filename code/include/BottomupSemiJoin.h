#pragma once
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;
using Table = vector<vector<int>>;

struct DOPaddingStats
{
    long long exactRows = 0;
    long long sensitivity = 1;
    int protectedRows = 0;
    long long protectedRowsExact = 0;
    long long paddingRowsExact = 0;
    double factor = 0.0;
    bool rrsAvailable = false;
    bool rrsTooExpensive = false; // RRS exceeds the budget at this (epsilon, delta).
    long long rrsSEST = 0;
    long long rrsSRRS = 0;
    long long rrsSHybrid = 0;
    long long rrsGS = 0;
    long long rrsMaxK = 0;
    double rrsEstLeaves = 0.0;
    long long rrsMaxEstTE = 0;
    int rrsMaxEstSubset = 0;
    long long rrsMaxRRSTE = 0;
    int rrsMaxRRSSubset = 0;
    long long rrsMaxHybridTE = 0;
    int rrsMaxHybridSubset = 0;
};

// Bottom-up semi-join; tables[i] is node i and parent[root] = -1.
// joinColInParent[i] and joinColInChild[i] identify the two join columns
// for edge (parent[i], i); neither is used for the root.
// protectedOutputRows > 0 is a public target >= exact output size. Zero uses
// the public exact output size. Empty roots require public rootColumns.
Table bottomUpSemiJoin(vector<Table> &tables, const vector<int> &parent, int root,
                      const vector<int> &joinColInParent, const vector<int> &joinColInChild,
                      int protectedOutputRows = 0, int rootColumns = 0);

long long computeAcyclicJoinOutputSize(vector<Table> tables, const vector<int> &parent, int root,
                                       const vector<int> &joinColInParent, const vector<int> &joinColInChild);

DOPaddingStats computeDOPaddingStats(vector<Table> tables, const vector<int> &parent, int root,
                                     const vector<int> &joinColInParent, const vector<int> &joinColInChild,
                                     int explicitOutputRows = 0,
                                     double epsilon = 1.0,
                                     double delta = 1e-9);

int computeDOProtectedOutputSize(vector<Table> tables, const vector<int> &parent, int root,
                                 const vector<int> &joinColInParent, const vector<int> &joinColInChild,
                                 int explicitOutputRows = 0,
                                 double epsilon = 1.0,
                                 double delta = 1e-9);

double getLastBottomUpParallelMs();
double getLastBottomUpSemiJoinMs();
double getLastBottomUpRootExpandMs();
