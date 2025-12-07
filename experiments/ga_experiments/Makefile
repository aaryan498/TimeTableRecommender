.PHONY: help install test quick-run full-run clean docker-build docker-run

help:
	@echo "GA Timetable Experiment Runner - Make targets:"
	@echo ""
	@echo "  install        - Install Python dependencies"
	@echo "  test           - Run unit tests"
	@echo "  quick-run      - Run quick test (20 runs, ~5 min)"
	@echo "  sample-data    - Generate sample dataset (100 runs, ~30 min)"
	@echo "  full-run       - Run full experiment (1000 runs, ~3-5 hours)"
	@echo "  clean          - Clean output directory"
	@echo "  docker-build   - Build Docker container"
	@echo "  docker-run     - Run experiments in Docker"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements-experiments.txt
	pip install -r ../../FinalScheduler/requirements.txt
	@echo "Done!"

test:
	@echo "Running unit tests..."
	python runner_tests.py
	@echo "All tests passed!"

quick-run:
	@echo "Running quick test (20 runs)..."
	python run_experiments.py \
		--config config/samples/quick_test_config.yaml
	@echo "Quick test complete! Check output/ga_quick_test.csv"

sample-data:
	@echo "Generating sample dataset (100 runs)..."
	python run_experiments.py \
		--config config/samples/experiment_config.yaml \
		--max-runs 100
	@echo "Sample dataset complete! Check output/ga_experiments_dataset.csv"

full-run:
	@echo "Running full experiment (1000+ runs)..."
	python run_experiments.py \
		--config config/samples/experiment_config.yaml \
		--max-runs 1000 \
		--workers 4
	@echo "Full experiment complete! Check output/ga_experiments_dataset.csv"

clean:
	@echo "Cleaning output directory..."
	rm -rf output/*.csv
	rm -rf output/*.parquet
	rm -rf output/raw/*
	rm -rf output/configs/*
	@echo "Output directory cleaned!"

docker-build:
	@echo "Building Docker container..."
	cd ../.. && docker build -t ga-experiments -f experiments/ga_experiments/Dockerfile .
	@echo "Docker image built: ga-experiments"

docker-run:
	@echo "Running experiments in Docker..."
	docker run -v $(PWD)/output:/app/experiments/ga_experiments/output \
		ga-experiments \
		--config config/samples/experiment_config.yaml \
		--max-runs 100
	@echo "Docker run complete!"

# Analysis helpers
analyze:
	@echo "Opening Python for analysis..."
	@echo "import pandas as pd"
	@echo "df = pd.read_csv('output/ga_experiments_dataset.csv')"
	@echo "print(df.head())"
	python -i -c "import pandas as pd; df = pd.read_csv('output/ga_experiments_dataset.csv'); print(f'Loaded {len(df)} rows')"

stats:
	@echo "Dataset statistics:"
	python -c "import pandas as pd; df = pd.read_csv('output/ga_experiments_dataset.csv'); print(f'Total runs: {len(df)}'); print(f'Success rate: {(df[\"process_exit_status\"] == \"success\").mean():.1%}'); print(f'Avg fitness: {df[\"fitness_final\"].mean():.3f}'); print(f'Avg runtime: {df[\"runtime_seconds\"].mean():.1f}s')"
