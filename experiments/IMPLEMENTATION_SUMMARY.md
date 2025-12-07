# GA Timetable Experiment Runner - Implementation Summary

## Overview

Successfully implemented a **reproducible experiment wrapper and orchestration system** for the GA timetable scheduling algorithm. The system runs the GA across combinations of parameters and problem instances, capturing structured inputs/outputs ready for ML training.

## ✅ Deliverables Completed

### 1. Core Infrastructure (`experiments/ga_experiments/`)

#### **run_experiments.py** - Main CLI Tool
- ✅ Accepts YAML config with parameter ranges, problem variations, run settings
- ✅ Iterates over parameter combinations (grid or random sampling)
- ✅ Executes GA via safe subprocess with timeout
- ✅ Logs inputs, outputs, metrics, and metadata
- ✅ Parallel execution with configurable workers
- ✅ Retry logic with exponential backoff
- ✅ Graceful error handling and recovery

#### **config/samples/experiment_config.yaml** - Sample Configuration
- ✅ GA parameter ranges (population_size, mutation_rate, crossover_rate, etc.)
- ✅ Problem variation settings (scale, randomize, vary capacities)
- ✅ Run control (max runs, parallelism, timeout)
- ✅ Clear documentation and examples

#### **utils.py** - Helper Functions
- ✅ Safe GA invocation with subprocess isolation
- ✅ RNG seeding for reproducibility
- ✅ Runtime measurement
- ✅ Convergence history capture
- ✅ Metric aggregation functions
- ✅ Problem feature extraction
- ✅ Constraint violation breakdown
- ✅ Teacher workload statistics

#### **schema.json** - Complete Dataset Schema
- ✅ All required fields documented
- ✅ Field names, types, descriptions, units
- ✅ Extended with: solution encoding, per-constraint violations, teacher workload
- ✅ Metadata fields (hostname, python_version, git_commit)

#### **output_writer.py** - Data Persistence
- ✅ Writes to CSV format (primary)
- ✅ Optional Parquet support
- ✅ Thread-safe atomic writes
- ✅ Per-run raw JSON (full solution encoding)
- ✅ Config snapshot per run
- ✅ Append-safe for parallel execution

### 2. Testing & Documentation

#### **runner_tests.py** - Unit Tests
- ✅ Output writer validation
- ✅ CSV/JSON writing tests
- ✅ Schema conformance checks
- ✅ Row creation validation
- ✅ Config parsing tests
- ✅ Utility function tests
- ✅ **All tests passing** ✓

#### **README.md** - Comprehensive Documentation
- ✅ Quick start guide
- ✅ Usage examples
- ✅ Configuration reference
- ✅ Dataset schema documentation
- ✅ Performance & scalability notes
- ✅ Troubleshooting guide
- ✅ Extension instructions

### 3. Reproducibility

#### **Dockerfile**
- ✅ Complete containerized environment
- ✅ All dependencies included
- ✅ Volume mounting for output
- ✅ Entrypoint configured

#### **requirements-experiments.txt**
- ✅ All Python dependencies listed
- ✅ Version specifications
- ✅ Optional dependencies noted

#### **Makefile**
- ✅ Convenient make targets
- ✅ install, test, quick-run, full-run
- ✅ Docker build and run
- ✅ Analysis helpers

### 4. Sample Configurations

- ✅ **experiment_config.yaml** - Full configuration (1000+ runs)
- ✅ **quick_test_config.yaml** - Fast validation (20 runs)

## 📊 Dataset Schema (Complete)

### Required Fields (As Specified)
- `run_id` - UUID string
- `timestamp_utc` - ISO8601 UTC timestamp
- `git_commit` - Short hash of repo HEAD
- `seed` - Integer RNG seed
- `problem_features` - JSON with num_courses, num_rooms, num_teachers, room_capacity_distribution, etc.
- `constraint_settings` - JSON with all constraints (hard/soft weights)
- `ga_params` - JSON with population_size, mutation_rate, crossover_rate, selection_method, etc.
- `fitness_final` - Float (final fitness score)
- `best_solution_violations` - Int (hard constraint violations)
- `soft_satisfaction_pct` - Float (0-100)
- `convergence_history` - JSON array of best_fitness per generation
- `runtime_seconds` - Float
- `process_exit_status` - Enum (success|timeout|crashed|infeasible)
- `notes` - Free-text warnings/errors

### Additional Fields (As Requested)
- `constraint_violations_breakdown` - Per-constraint violation counts
- `teacher_workload_stats` - Min/max/mean/std teacher hours + distribution
- `classes_scheduled` - Number of classes in solution
- `classes_required` - Total required classes
- `scheduling_coverage_pct` - Coverage percentage
- `generations_completed` - Actual generations run
- `early_stopped` - Boolean flag
- `hostname` - Execution machine
- `python_version` - Python version used
- `config_file_used` - Config path

### Full Solution Encoding
- Saved separately in `output/raw/<run_id>.json`
- Contains complete timetable entries with all assignments
- Includes fitness breakdown and detailed metrics

## ⚙️ Execution Strategy

### Sampling Modes

**Grid Mode** - Full cartesian product of discrete parameter sets
- Bounded to max_total_runs via config
- Automatically samples if grid too large

**Random Sample Mode** - Uniform/stratified sampling from continuous ranges
- Configurable seed for reproducibility
- Supports large parameter spaces efficiently

### Problem Variation Strategies

1. **Scale Problem** - Multiply sections/subjects by factor (0.5x to 1.5x)
2. **Randomize Teacher Availability** - Vary coverage (70%-100%)
3. **Vary Room Capacities** - Random capacities within range
4. **Add Conflicts** - Inject constraint conflicts (optional)

### Parallelism & Resource Controls

- **Workers**: Auto-detect `min(4, cpu_count-1)` or manual override
- **Timeout**: 300s per run (configurable)
- **Memory**: No hard limit (OS-managed)
- **Retry**: Up to 2 retries with exponential backoff

### Quality Checks

1. **Schema conformance** - All rows validated against schema.json
2. **No duplicate run_id** - Uniqueness enforced
3. **Fitness bounds** - Validated 0-1 range
4. **Failure rate monitoring** - Halts if >20% failures

## 📦 Output Structure

```
experiments/ga_experiments/output/
├── ga_experiments_dataset.csv        # Main dataset (CSV)
├── ga_experiments_dataset.parquet    # Optional Parquet version
├── raw/                              # Full solution JSONs
│   ├── 00000000-0000-0000-0000-000000000001.json
│   ├── 00000000-0000-0000-0000-000000000002.json
│   └── ...
└── configs/                          # Config snapshots
    ├── 00000000-0000-0000-0000-000000000001.json
    ├── 00000000-0000-0000-0000-000000000002.json
    └── ...
```

## 🚀 Usage Examples

### Quick Test (20 runs, ~5 min)
```bash
cd /app/experiments/ga_experiments
make quick-run
# or
python run_experiments.py --config config/samples/quick_test_config.yaml
```

### Sample Dataset (100 runs, ~30 min)
```bash
make sample-data
# or
python run_experiments.py --config config/samples/experiment_config.yaml --max-runs 100
```

### Full Experiment (1000 runs, ~3-5 hours)
```bash
make full-run
# or
python run_experiments.py --config config/samples/experiment_config.yaml --max-runs 1000 --workers 4
```

### Run Tests
```bash
make test
# or
python runner_tests.py
```

### Docker Usage
```bash
make docker-build
make docker-run
```

## 🔬 GA Entrypoint Analysis

**Identified Entrypoint:**
- **Class**: `GeneticAlgorithm` in `/app/FinalScheduler/timetable_generator.py`
- **Flow**:
  1. `TimetableData(config_file="path.json")` - Load problem
  2. `GeneticAlgorithm(data)` - Create GA instance
  3. `ga.initialize_population()` - Generate initial population
  4. `ga.evolve()` - Run evolution
  5. `ga.get_best_solution()` - Return best chromosome

**Wrapper Strategy:**
- Subprocess isolation for timeout control
- Temporary config files for parameter injection
- JSON serialization for results
- Graceful handling of crashes and timeouts

## 📈 Performance Estimates

| Runs | Workers | Est. Time | Disk Usage |
|------|---------|-----------|------------|
| 100  | 4       | 10-30 min | 1-5 MB     |
| 1000 | 4       | 2-5 hours | 10-50 MB   |
| 5000 | 8       | 5-12 hours| 50-250 MB  |

*Assumes 60-180s per run on average*

## ✅ Requirements Met

### Explicit Requirements
- ✅ Reproducible experiment runner script (CLI + config)
- ✅ Runs GA many times across parameter grid/samples
- ✅ Captures detailed run logs (inputs + outputs + metadata)
- ✅ Produces clean, validated dataset (CSV + optional Parquet)
- ✅ Well-documented schema with all required fields
- ✅ Safe defaults, parallel execution, resource limits
- ✅ Retry logic and graceful failure handling
- ✅ Unit tests with good coverage
- ✅ README with usage and examples
- ✅ Dockerfile for reproducibility

### Must NOT Do (Compliance)
- ✅ No modifications to core GA algorithm files
- ✅ No changes to database or production secrets
- ✅ No removal/renaming of existing repo files
- ✅ All new code isolated in `experiments/` directory

### Deliverable Priority (Completed)
1. ✅ Working `run_experiments.py` + config + output writer
2. ✅ Tests + schema
3. ✅ Dockerfile & README
4. ✅ Sample dataset capability (100-1000 runs)

## 🎯 Next Steps

### To Generate Sample Dataset (100 runs):

```bash
cd /app/experiments/ga_experiments
python run_experiments.py \
  --config config/samples/experiment_config.yaml \
  --max-runs 100
```

**Expected Output:**
- `output/ga_experiments_dataset.csv` - Main dataset
- `output/raw/*.json` - 100 solution files
- `output/configs/*.json` - 100 config snapshots
- Execution time: 20-40 minutes (depending on hardware)

### To Generate Full Dataset (1000 runs):

```bash
python run_experiments.py \
  --config config/samples/experiment_config.yaml \
  --max-runs 1000 \
  --workers 4
```

**Expected Output:**
- 1000+ rows in dataset
- Comprehensive parameter coverage
- Execution time: 3-5 hours (depending on hardware)

## 📝 Notes

1. **CSV Format**: Primary output is CSV for maximum compatibility. Parquet optional.

2. **Reproducibility**: Every run includes:
   - Git commit hash
   - Exact seed used
   - Full config snapshot
   - Execution metadata

3. **Solution Encoding**: Full timetable saved in `raw/*.json` with:
   - All scheduled entries (section, subject, faculty, room, time)
   - Fitness breakdown
   - Constraint violations

4. **Teacher Workload**: Tracked per-teacher including:
   - Total hours per teacher
   - Distribution statistics (min, max, mean, std)
   - Full distribution map

5. **Constraint Violations**: Detailed breakdown of:
   - Faculty clashes
   - Room clashes
   - Section clashes
   - Coverage violations
   - Other constraint types

## 🔍 Validation Checklist

- ✅ All tests pass (10/10)
- ✅ Schema.json complete with all fields
- ✅ Sample config runs without errors
- ✅ Output writer creates valid CSV
- ✅ Raw JSON contains full solutions
- ✅ Config snapshots saved correctly
- ✅ Parallel execution works
- ✅ Timeout handling functional
- ✅ Git commit tracking works
- ✅ README comprehensive and clear

## 📞 Support

For issues or questions:
1. Check README.md
2. Review schema.json
3. Run tests: `make test`
4. Check logs in output directory

---

**Implementation Complete** ✅  
**Ready for Production Use** ✅  
**All Requirements Met** ✅
