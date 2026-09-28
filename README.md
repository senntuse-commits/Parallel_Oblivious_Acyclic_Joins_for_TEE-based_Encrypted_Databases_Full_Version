# Parallel Oblivious Acyclic Joins for TEE-based Encrypted Databases (Full Version)

**Paper:** [Full version PDF](full-version.pdf)

JFYan is a parallel oblivious join-free Yannakakis framework for acyclic joins.
It computes subtree contributions bottom-up and materializes the output through
top-down position propagation. This repository provides the Intel SGX
implementation and benchmark scripts for TPC-DS, TPC-H, IMDb, and Email-EuAll.

## Algorithms

| Option | Algorithm |
|---|---|
| `-JFYan` | JFYan: contribution computation and position propagation. This is the default. |
| `-ParYan` | ParYan: parallel oblivious Yannakakis baseline. |
| `-ObliYan` | ObliYan: oblivious Yannakakis baseline. |
| `-NonObliJFYan` | Non-oblivious reference implementation of JFYan. |

`--all` runs JFYan, ParYan, and ObliYan in sequence.

## Source Attribution

The residual-sensitivity calculation in
[`RRSSensitivity.h`](code/include/RRSSensitivity.h) is adapted from
[DOJoin](https://github.com/z46wu/DOJoin)
([original calculator](https://github.com/z46wu/DOJoin/blob/main/code%20and%20data/rs_calculator.h))
and is used in the benchmark
target-size simulation. For DO execution, JFYan can serve as the parallel
oblivious join evaluator within that framework, using public targets supplied
by its protected advice-generation procedure.

The bundled [TPC-H](https://www.tpc.org/tpch/) and
[TPC-DS](https://www.tpc.org/tpcds/) data generators retain their original
documentation and notices. The TPC-H
[DBGEN/QGEN README](https://github.com/electrum/tpch-dbgen/blob/master/README)
is available in a public mirror and has the same text as
[`code/tpch/dbgen/README`](code/tpch/dbgen/README). The TPC-DS license is in
[`code/tpcds/EULA.txt`](code/tpcds/EULA.txt).

## Repository layout

| Path | Contents |
|---|---|
| [`full-version.pdf`](full-version.pdf) | Full paper, proofs, and extended evaluation. |
| [`code/App/`](code/App/) | Host application: input loading, command-line options, and benchmark output. |
| [`code/Enclave/`](code/Enclave/) | Enclave entry points, configuration, and execution. |
| [`code/implement/`](code/implement/), [`code/include/`](code/include/) | Join algorithms, oblivious primitives, and shared headers. |
| [`code/tpcds/`](code/tpcds/), [`code/tpch/`](code/tpch/) | Data generators and query-specific input preparation. |
| [`code/experiments/`](code/experiments/) | Workload preparation and experiment runners. |
| [`code/tests/`](code/tests/) | Native join-correctness tests. |
| [`code/exec.sh`](code/exec.sh) | Build-and-run script. |

## Requirements

| Component | Requirement |
|---|---|
| Platform | Linux x86-64. |
| Processor | Intel CPU and platform supporting [SGX2/EDMM](https://www.intel.com/content/www/us/en/support/articles/000058764/software/intel-security-products.html), with SGX enabled in BIOS/UEFI, for hardware experiments. |
| Memory | System RAM and enclave page cache (EPC) capacity appropriate for the selected workload. |
| SGX | Intel SGX driver/runtime and SDK with EDMM support, including trusted OpenMP and pthread libraries (`sgx_omp`, `sgx_pthread`). |
| Compiler | GCC/G++ with C++17 support. |
| Build tools | CMake 3.15+, Make, and OpenSSL. |
| Scripts | Bash and Python 3.10+. |

The main experiments ran on an **Intel Xeon Platinum 8369B @ 2.70 GHz**, using
**16 physical cores** (one worker per core) and **64 GB EPC**. Additional
experiments used **32 GB EPC** to evaluate paging. These are the paper's
experimental configurations; memory requirements depend on the workload.

The default SDK location is `/opt/intel/sgxsdk`. Set `SGX_SDK` if it is installed
elsewhere. Enclave heap and thread limits are configured in
[`Enclave.config.xml`](code/Enclave/Enclave.config.xml).

## Build and run

From the repository root, build and run the included small example:

```bash
cd code
source /opt/intel/sgxsdk/environment
./exec.sh -JFYan --stage-profile -t 16
```

All commands below run from `code/`. The script builds the host application and
signed enclave before execution. The signing key is generated locally on the
first build.

To build manually:

```bash
cmake -S . -B build -DSGX_SDK="${SGX_SDK:-/opt/intel/sgxsdk}" \
  -DSGX_MODE=HW -DSGX_DEBUG=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
./build/App/app -JFYan --stage-profile -t 16
```

To use the SGX simulator, run `SGX_MODE=SIM ./exec.sh -JFYan --stage-profile`.
Use hardware mode for SGX performance measurements.

## Prepare data and run experiments

### TPC-DS

Generate the Q18 and Q85 inputs at scale factor 1, then compare the three
oblivious algorithms:

```bash
python3 tpcds/prepare_query18_scales.py --scales 1
python3 tpcds/prepare_query85_scales.py --scales 1

./exec.sh --sql18 tpcds/sql18_projected/sf_1 --all --stage-profile -t 16
./exec.sh --sql85 tpcds/sql85_projected/sf_1 --all --stage-profile -t 16
```

Pass a comma-separated list to prepare several scales, for example
`--scales 1,5,10,20,30`. The preparation scripts generate the required tables and
their integer projections. Projected input directories include the relation
files, `expected.txt`, and `stats.txt`.

### TPC-H Q9

Prepare Q9 at scale factor 0.1 and run with a public materialization target:

```bash
./experiments/run_experiments.sh prepare-tpch
./exec.sh --tpch9 tpch/sf_0p1 --public-target 15014300 --all -t 16
```

For an exact-size timing experiment, use `--no-materialize-padding` in place
of `--public-target 15014300` and add `--stage-profile`.

### Real datasets and join-tree workloads

The experiment runner prepares the JOB 13d join core from IMDb and join trees
from Email-EuAll:

```bash
./experiments/run_experiments.sh prepare-job
./experiments/run_experiments.sh job13d

./experiments/run_experiments.sh prepare-snap
./experiments/run_experiments.sh snap
```

See [the experiment instructions](code/experiments/README.md) for Q85 chain
and star workloads, input scales, tree shapes, and memory measurements.
Generated datasets are stored locally and are not included in the repository.

### Threads and repeated runs

```bash
./exec.sh --sql85 tpcds/sql85_projected/sf_1 -JFYan \
  --stage-profile --thread-sweep 1,2,4,8,16

SCALE=1 THREADS=16 REPEATS=3 ./experiments/run_experiments.sh q18-jfyan
```

The runner writes logs to `experiments/results/`. Set `RESULTS_DIR` to choose
another location.

## Command-line options

| Option | Meaning |
|---|---|
| `--sql18 DIR`, `--sql85 DIR` | Load a projected TPC-DS query. |
| `--sql85-chain3 DIR`, `--sql85-returns-star DIR` | Load a Q85 chain or star workload. |
| `--tpch9 DIR` | Load prepared TPC-H Q9 input; supply a public target or select exact-size execution. |
| `--tree-workload DIR` | Load a prepared generic join tree, including the IMDb and Email-EuAll workloads. |
| `-t N` | Number of enclave OpenMP threads; default `16`. |
| `--thread-sweep LIST` | Run several thread counts in the same enclave. |
| `--bench-only` | Return result dimensions without copying the result tuples to the host. |
| `--stage-profile` | Report stage timings; also enables `--bench-only`. |
| `--profile` | Include detailed enclave and primitive timing output. |
| `--memory-profile` | Track C++ heap allocations; invoke through `exec.sh` to enable the instrumentation. |
| `--public-target N` | Execute using an externally supplied public output-size bound. |
| `--no-materialize-padding` | Select exact-size benchmark execution. |
| `-tau N` | Set the allocation target for a benchmark, or pair with `--materialize-padding` to supply a public target. |
| `-m N` | Result-copy buffer capacity in integer cells; default `1000000`. |
| `--print-limit N` | Maximum result rows printed; default `20`, or `0` to print none. |
| `--help` | Show the available options. |

Choose one dataset option per run. Without one, the program uses the built-in
example. Public targets must cover the output size. The default output-row
limit is 200,000,000; set `JFYAN_MAX_MATERIALIZED_ROWS` to override it within
`1..268435455`.

The public-target interface consumes advice supplied externally.
`--benchmark-target-simulation` is a separate deterministic sizing experiment;
its `--do-epsilon` and `--do-delta` options do not generate a DP release.

## Reading the output

`Result` gives the returned row and column counts. `ECALL time` measures the
host-side enclave call; `Join-only time` reports the join algorithm stages.
Times are in milliseconds. Stage profiling separates filtering and
materialization, while detailed profiling also reports oblivious primitives.

Memory profiling reports tracked C++ heap allocations and includes
instrumentation overhead. EPC capacity is a machine setting; the runner's
`EPC_CONFIG_GB` variable records that setting in the log.

## Native tests

The correctness tests run without SGX using a C++17 compiler and OpenMP:

```bash
cmake -S tests -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j
ctest --test-dir build-native --output-on-failure
```

The tests compare join results with exhaustive enumeration and cover padding,
empty inputs, deep trees, and extreme keys.
