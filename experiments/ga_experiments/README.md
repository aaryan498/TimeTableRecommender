# GA Timetable Scheduling Experiment Runner

A reproducible experiment framework for running the Genetic Algorithm (GA) timetable scheduler across combinations of problem instances and GA parameter settings. Generates high-quality datasets ready for machine learning training.

## Overview

This experiment runner:
- ✅ Executes GA across parameter grids or random samples
- ✅ Varies problem instances (courses, teachers, rooms, constraints)
- ✅ Captures comprehensive inputs and outputs for ML training
- ✅ Handles timeouts, retries, and failures gracefully
- ✅ Supports parallel execution with configurable workers
- ✅ Produces validated CSV/Parquet datasets with full metadata
- ✅ Saves full solution encodings as JSON for detailed analysis
- ✅ Tracks convergence history and constraint violations

## Quick Start

### Installation

```bash
# Install Python dependencies
pip install -r /app/FinalScheduler/requirements.txt
pip install pyyaml

# Optional: For Parquet support
pip install pandas pyarrow
```

### Run Sample Experiment (100 runs)

```bash
cd /app
python experiments/ga_experiments/run_experiments.py \
  --config experiments/ga_experiments/config/samples/experiment_config.yaml \
  --max-runs 100
```

### Generate 1000+ Runs

```bash
python experiments/ga_experiments/run_experiments.py \
  --config experiments/ga_experiments/config/samples/experiment_config.yaml \
  --max-runs 1000 \
  --workers 4
```

## Directory Structure

```
experiments/ga_experiments/
├── run_experiments.py          # Main CLI runner
├── utils.py                    # Helper functions (seeding, metrics, GA invocation)
├── output_writer.py            # CSV/Parquet/JSON output handling
├── runner_tests.py             # Unit tests
├── schema.json                 # Complete dataset schema documentation
├── README.md                   # This file
├── Dockerfile                  # Reproducibility container
├── requirements-experiments.txt # Python dependencies
├── config/
│   └── samples/
│       └── experiment_config.yaml  # Sample configuration
└── output/
    ├── ga_experiments_dataset.csv  # Main dataset
    ├── raw/                        # Full solution JSONs (one per run)
    │   ├── <run_id_1>.json
    │   └── <run_id_2>.json
    └── configs/                    # Config snapshots (one per run)
        ├── <run_id_1>.json
        └── <run_id_2>.json
```

## Configuration

### Experiment Config (`experiment_config.yaml`)

Key sections:

#### 1. Run Settings
```yaml
run_settings:
  max_total_runs: 1000           # Maximum number of runs
  sampling_mode: "random"        # "grid" or "random"
  repetitions_per_config: 2      # Repeats per unique config
```

#### 2. GA Parameter Ranges
```yaml
ga_parameters:
  population_size:
    type: "discrete"
    values: [30, 50, 75, 100, 150]
  
  mutation_rate:
    type: "continuous"
    min: 0.05
    max: 0.4
  
  generations:
    type: "discrete"
    values: [50, 100, 150, 200]
```

#### 3. Problem Instance Variation
```yaml
problem_instance:
  base_config: "FinalScheduler/corrected_timetable_config.json"
  
  variations:
    scale_problem:
      enabled: true
      scale_factors: [0.5, 0.75, 1.0, 1.25]
    
    randomize_teacher_availability:
      enabled: true
      coverage_range:
        min: 0.7
        max: 1.0
```

#### 4. Resource Limits
```yaml
resource_limits:
  parallel_workers: "auto"        # "auto" or integer
  per_run_timeout_seconds: 300    # 5 minutes
  max_retries: 2
```

## Dataset Schema

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique UUID for run |
| `timestamp_utc` | string | ISO8601 UTC timestamp |
| `git_commit` | string | Git commit hash (reproducibility) |
| `seed` | integer | RNG seed used |
| `problem_features` | object | Problem instance characteristics |
| `ga_params` | object | GA parameter settings |
| `fitness_final` | float | Final fitness score |
| `best_solution_violations` | integer | Hard constraint violations |
| `convergence_history` | string | JSON array of fitness per generation |
| `runtime_seconds` | float | Execution time |
| `process_exit_status` | enum | success/timeout/crashed/infeasible |

### Extended Fields

| Field | Type | Description |
|-------|------|-------------|
| `constraint_violations_breakdown` | object | Per-constraint violation counts |
| `teacher_workload_stats` | object | Workload distribution statistics |
| `classes_scheduled` | integer | Number of classes scheduled |
| `classes_required` | integer | Number of classes required |
| `scheduling_coverage_pct` | float | Percentage coverage |
| `early_stopped` | boolean | Whether GA stopped early |

See `schema.json` for complete field definitions.

## Output Formats

### 1. CSV Dataset
- **File**: `output/ga_experiments_dataset.csv`
- **Format**: Standard CSV with headers
- **Usage**: Easy to load in Pandas, Excel, R
- **Note**: Complex objects stored as JSON strings

### 2. Parquet Dataset (optional)
- **File**: `output/ga_experiments_dataset.parquet`
- **Format**: Apache Parquet columnar format
- **Usage**: Efficient for large datasets, Spark, Dask
- **Requires**: `pip install pandas pyarrow`

### 3. Raw Solution JSONs
- **Directory**: `output/raw/`
- **Format**: One JSON file per run
- **Content**: Full timetable solution encoding
- **Usage**: Detailed analysis, visualization, debugging

### 4. Config Snapshots
- **Directory**: `output/configs/`
- **Format**: One JSON file per run
- **Content**: Exact configuration used for run
- **Usage**: Reproducibility, parameter inspection

## Usage Examples

### Basic Usage

```bash
# Run with default config
python run_experiments.py --config config/samples/experiment_config.yaml

# Override max runs
python run_experiments.py --config config.yaml --max-runs 500

# Override workers
python run_experiments.py --config config.yaml --workers 8

# Dry run (validate config)
python run_experiments.py --config config.yaml --dry-run
```

### Load Results in Python

```python
import pandas as pd
import json

# Load CSV dataset
df = pd.read_csv('experiments/ga_experiments/output/ga_experiments_dataset.csv')

print(f"Total runs: {len(df)}")
print(f"Success rate: {(df['process_exit_status'] == 'success').mean():.1%}")
print(f"Avg fitness: {df['fitness_final'].mean():.3f}")

# Parse JSON fields
df['ga_params_parsed'] = df['ga_params'].apply(json.loads)
df['problem_features_parsed'] = df['problem_features'].apply(json.loads)

# Analyze convergence
df['convergence'] = df['convergence_history'].apply(json.loads)
df['generations_to_converge'] = df['convergence'].apply(len)

# Load raw solution
with open('output/raw/<run_id>.json', 'r') as f:
    solution = json.load(f)
```

### Analyze Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Fitness vs population size
sns.boxplot(data=df, x='population_size', y='fitness_final')
plt.title('Fitness by Population Size')
plt.show()

# Runtime vs generations
plt.scatter(df['generations_completed'], df['runtime_seconds'])
plt.xlabel('Generations Completed')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs Generations')
plt.show()

# Success rate by parameter
success_by_mutation = df.groupby('mutation_rate')['process_exit_status'].apply(
    lambda x: (x == 'success').mean()
)
success_by_mutation.plot(kind='bar')
plt.title('Success Rate by Mutation Rate')
plt.ylabel('Success Rate')
plt.show()
```

## Testing

Run unit tests:

```bash
python experiments/ga_experiments/runner_tests.py
```

Tests cover:
- ✅ Output writer functionality
- ✅ CSV/JSON writing
- ✅ Row creation and validation
- ✅ Schema compliance
- ✅ Configuration parsing
- ✅ Utility functions

## Docker Usage

Build container:

```bash
cd /app/experiments/ga_experiments
docker build -t ga-experiments .
```

Run experiments:

```bash
docker run -v $(pwd)/output:/output ga-experiments \
  --config /app/config/samples/experiment_config.yaml \
  --max-runs 1000
```

## Performance & Scalability

### Resource Usage

- **CPU**: Auto-detects cores, uses `min(4, cpu_count-1)` by default
- **Memory**: No hard limit, relies on OS management
- **Disk**: ~1-5 KB per run (CSV) + ~10-50 KB per run (raw JSON)

### Estimated Runtimes

| Runs | Workers | Est. Time | Disk Usage |
|------|---------|-----------|------------|
| 100  | 4       | 10-30 min | 1-5 MB     |
| 1000 | 4       | 2-5 hours | 10-50 MB   |
| 5000 | 8       | 5-12 hours| 50-250 MB  |

*Assumes 60-180s per run on average*

### Parallelism

- Uses `ProcessPoolExecutor` for true parallel execution
- Each run isolated in subprocess with timeout
- Safe for multi-core machines
- Configurable worker count

## Validation & Quality Checks

### Automatic Validation

1. **Schema conformance**: All rows validated against `schema.json`
2. **Duplicate detection**: Ensures unique `run_id` for each run
3. **Fitness bounds**: Validates fitness in plausible range (0-1)
4. **Failure rate monitoring**: Halts if failure rate exceeds threshold (default 20%)

### Manual Validation

```python
import pandas as pd
import json

df = pd.read_csv('output/ga_experiments_dataset.csv')

# Check for duplicates
assert df['run_id'].is_unique, "Duplicate run_ids found!"

# Check fitness bounds
assert (df['fitness_final'] >= 0).all(), "Negative fitness found!"
assert (df['fitness_final'] <= 1).all(), "Fitness > 1 found!"

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Check exit status distribution
print("\nExit status distribution:")
print(df['process_exit_status'].value_counts())
```

## Troubleshooting

### Common Issues

**1. Timeout errors**
- **Cause**: GA taking too long
- **Solution**: Increase `per_run_timeout_seconds` in config or reduce `generations`

**2. High crash rate**
- **Cause**: Problem instances too large or infeasible
- **Solution**: Scale down problem sizes or check constraint conflicts

**3. Out of memory**
- **Cause**: Too many parallel workers
- **Solution**: Reduce `parallel_workers` (e.g., `--workers 2`)

**4. Slow execution**
- **Cause**: Sequential execution or low CPU
- **Solution**: Increase workers or use more powerful machine

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extending the Framework

### Add New GA Parameters

1. Edit `config/samples/experiment_config.yaml`:
```yaml
ga_parameters:
  your_new_param:
    type: "discrete"
    values: [10, 20, 30]
```

2. Update schema in `schema.json` if needed

### Add New Problem Variations

1. Implement variation function in `run_experiments.py`
2. Add configuration in YAML:
```yaml
variations:
  your_new_variation:
    enabled: true
    param: value
```

### Add New Output Fields

1. Update `schema.json`
2. Modify `create_row_from_run_result()` in `output_writer.py`
3. Extract data in `utils.py`

## Citation

If you use this experiment framework in your research, please cite:

```bibtex
@software{ga_timetable_experiments,
  title={GA Timetable Scheduling Experiment Runner},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## License

Same license as the parent GA timetable scheduler project.

## Support

For issues or questions:
1. Check this README
2. Review `schema.json` for field definitions
3. Run tests: `python runner_tests.py`
4. Check logs in `experiments/ga_experiments/experiment_run.log`

## Version History

- **v1.0** (2025-01-XX): Initial release
  - Grid and random sampling modes
  - Problem instance variation
  - CSV/Parquet output
  - Full solution encoding
  - Parallel execution
  - Comprehensive tests
