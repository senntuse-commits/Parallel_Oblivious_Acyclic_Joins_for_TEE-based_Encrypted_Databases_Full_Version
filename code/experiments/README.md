# Experiments

Run commands from the parent `code` directory. Dataset generation and enclave
experiments require the Linux environment described in [README.md](../README.md).
Python 3.10+ is required for workload preparation. Generated datasets and run
logs are excluded from version control.

## Prepare workloads

```bash
python3 tpcds/prepare_query18_scales.py --scales 1,5,10,20,30
python3 tpcds/prepare_query85_scales.py --scales 1,5,10,20,30
python3 tpcds/prepare_query85_chain3_scales.py --scales 1,5,10,20,30
python3 tpcds/prepare_query85_returns_star_scales.py --scales 1,5,10,20,30
./experiments/run_experiments.sh prepare-tpch
./experiments/run_experiments.sh prepare-job
./experiments/run_experiments.sh prepare-job-scales
./experiments/run_experiments.sh prepare-snap
```

The TPC scripts build the included data generators when needed. JOB commands
download the IMDb data and prepare the JOB 13d join core, including five input
scales. SNAP prepares controlled Email-EuAll join trees. Large datasets are
downloaded/generated locally and are not included in this package.

## Run experiments

The defaults are 16 threads and one measured run. New logs go to
`experiments/results/`; `REPEATS`, `THREADS`, `SCALE`, and
`RESULTS_DIR` override these settings.

| Command | Workload |
|---|---|
| `benchmark` | TPC-DS Q18/Q85 and TPC-H Q9 |
| `job13d` / `job13d-scale` | Full JOB 13d or one selected input scale |
| `snap` / `snap-case` | All Email-EuAll trees or one selected tree |
| `nonoblivious` | JFYan and NonObliJFYan |
| `memory` | Tracked C++ heap measurements |
| `q18-jfyan` / `q85-jfyan` | One TPC-DS query/scale |
| `q18-nonobli` / `q85-nonobli` | Its non-oblivious reference |
| `q18-memory` / `q85-memory` | Its heap profile |
| `tpch9-target-simulation` | Deterministic benchmark target sizing |

```bash
SCALE=30 THREADS=16 ./experiments/run_experiments.sh q18-jfyan
JOB13D_PERCENT=60 ./experiments/run_experiments.sh job13d-scale
./exec.sh --sql85-chain3 tpcds/sql85_chain3_projected/sf_30 --all --profile -t 16
./exec.sh --sql85-returns-star tpcds/sql85_returns_star_projected/sf_30 --all --profile -t 16
```

Use `./experiments/run_experiments.sh --help` for all options. Memory profiling
uses a separate build and its times include instrumentation overhead. Changing
`EPC_CONFIG_GB` only labels a run; the actual EPC configuration must be set on
the machine.

TPC-H fixed-target profiling uses `--no-materialize-padding -tau N` to allow
benchmark telemetry at a chosen target. For the restricted public-target
interface, use `--public-target N`; obtain that target from the external
protected advice mechanism for a DO integration. The simulation options
`DO_EPSILON` and `DO_DELTA` do not implement that mechanism.

