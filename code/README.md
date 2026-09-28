# JFYan

Implementation and experiment scripts for [the full version](../full-version.pdf)
of *Parallel Oblivious Acyclic Joins for TEE-based Encrypted Databases*.

## Build and run

The enclave experiments require Linux x86-64, an Intel CPU and platform with
SGX2/EDMM support and SGX enabled in BIOS/UEFI, an EDMM-capable SGX driver/runtime,
and the Intel SGX SDK
(including its trusted OpenMP library), GCC/G++, CMake 3.15+, Make, and OpenSSL.
The main experiments ran on an Intel Xeon Platinum 8369B @ 2.70 GHz, using 16
physical cores and 64 GB EPC; additional paging experiments used 32 GB EPC. See the
[hardware requirements](../README.md#requirements) for details.

The enclave configuration allows up to 64 GiB of heap and 64 enclave threads;
the available EPC is determined by the machine configuration.

Run the following from this `code` directory:

```bash
chmod +x exec.sh experiments/run_experiments.sh
source /opt/intel/sgxsdk/environment
./exec.sh -JFYan --profile -t 16
```

This command builds and runs the built-in small example. Set `SGX_SDK` if the
SDK is installed elsewhere. CMake generates a local enclave signing key on the
first build; private keys are neither distributed nor tracked.

The public output-row limit defaults to 200,000,000.
Override it with `JFYAN_MAX_MATERIALIZED_ROWS=N ./exec.sh ...`
or the CMake option `-DJFYAN_MAX_MATERIALIZED_ROWS=N` (range 1--268,435,455).
Each input relation is also limited to 268,435,455 rows to keep merged sorting
workspaces within the implementation's integer-index range. Configure the
enclave heap and system RAM for the chosen workload.
`SGX_MODE=SIM ./exec.sh ...` selects SGX simulation mode.

## Native correctness tests

The regression tests can run without SGX using a C++17 compiler and
OpenMP:

```bash
cmake -S tests -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j
ctest --test-dir build-native --output-on-failure
```

These tests compare small join results with exhaustive enumeration and cover
JFYan padding, empty inputs, deep trees, extreme keys, and the three reference
implementations.

## Public materialization targets

| Option | Meaning |
|---|---|
| `--no-materialize-padding` | Exact-size benchmark mode; the output size is public. Optional `-tau N` fixes the benchmark allocation target. |
| `--public-target N` | Consume an externally released public target `Lambda = N`. |
| `--benchmark-target-simulation` | Deterministic benchmark target-size simulation. |

For `--public-target`, supply a bound covering the true output size and within
the configured row limit. This mode suppresses exact-cardinality telemetry and
plaintext result copy-out. Protected advice generation and encrypted client
transport are external components. `--do-epsilon` and `--do-delta` configure
only the deterministic benchmark simulation.

## Experiments

See [experiments/README.md](experiments/README.md) for workload preparation,
experiment commands, and runtime options.

## Third-party components

The residual-sensitivity calculation in [RRSSensitivity.h](include/RRSSensitivity.h)
is adapted from [DOJoin](https://github.com/z46wu/DOJoin).
The [TPC-H](https://www.tpc.org/tpch/) and [TPC-DS](https://www.tpc.org/tpcds/)
data generators retain their original documentation and notices; the TPC-DS
license is in [tpcds/EULA.txt](tpcds/EULA.txt). See
[Source Attribution](../README.md#source-attribution) for the original project
and documentation links.
