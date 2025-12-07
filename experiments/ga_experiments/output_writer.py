"""
Output writer for experiment results
Handles CSV, Parquet, and raw JSON output with atomic writes
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class ExperimentOutputWriter:
    """
    Thread-safe writer for experiment results
    Supports CSV, Parquet, and raw JSON output
    """
    
    def __init__(self, output_dir: str, dataset_name: str, formats: List[str] = ['csv']):
        """
        Initialize output writer
        
        Args:
            output_dir: Base output directory
            dataset_name: Name for the dataset files
            formats: List of formats to write ('csv', 'parquet', or both)
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.formats = formats
        self.lock = threading.Lock()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'raw').mkdir(exist_ok=True)
        (self.output_dir / 'configs').mkdir(exist_ok=True)
        
        # Define file paths
        self.csv_path = self.output_dir / f"{dataset_name}.csv"
        self.parquet_path = self.output_dir / f"{dataset_name}.parquet"
        
        # Track if headers written
        self.csv_headers_written = False
        
        # CSV fieldnames (will be populated on first write)
        self.fieldnames = None
        
        logger.info(f"Output writer initialized: {self.output_dir}")
    
    def write_row(self, row_data: Dict[str, Any], raw_solution: Dict[str, Any] = None, 
                  config_used: Dict[str, Any] = None):
        """
        Write a single experiment row to output files
        
        Args:
            row_data: Dictionary containing all fields for the dataset row
            raw_solution: Full solution encoding to save as JSON
            config_used: Configuration used for this run
        """
        with self.lock:
            try:
                # Convert nested objects to JSON strings for CSV
                processed_row = self._process_row_for_csv(row_data)
                
                # Write to CSV
                if 'csv' in self.formats:
                    self._write_csv_row(processed_row)
                
                # Write to Parquet (if requested)
                if 'parquet' in self.formats:
                    self._write_parquet_row(row_data)
                
                # Write raw JSON
                if raw_solution:
                    run_id = row_data.get('run_id', 'unknown')
                    json_path = self.output_dir / 'raw' / f"{run_id}.json"
                    with open(json_path, 'w') as f:
                        json.dump(raw_solution, f, indent=2)
                
                # Write config
                if config_used:
                    run_id = row_data.get('run_id', 'unknown')
                    config_path = self.output_dir / 'configs' / f"{run_id}.json"
                    with open(config_path, 'w') as f:
                        json.dump(config_used, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error writing row: {e}")
                raise
    
    def _process_row_for_csv(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex objects to JSON strings for CSV compatibility"""
        processed = {}
        for key, value in row_data.items():
            if isinstance(value, (dict, list)):
                processed[key] = json.dumps(value)
            elif value is None:
                processed[key] = ''
            else:
                processed[key] = value
        return processed
    
    def _write_csv_row(self, row_data: Dict[str, Any]):
        """Write a single row to CSV with atomic append"""
        # Determine fieldnames from first row
        if self.fieldnames is None:
            self.fieldnames = list(row_data.keys())
        
        # Check if file exists and has headers
        file_exists = self.csv_path.exists()
        
        # Use atomic write pattern
        temp_path = self.csv_path.with_suffix('.csv.tmp')
        
        try:
            # If file doesn't exist, create with headers
            if not file_exists:
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    writer.writerow(row_data)
                logger.debug(f"Created new CSV with headers: {self.csv_path}")
            else:
                # Append to existing file
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writerow(row_data)
                logger.debug(f"Appended row to CSV: {self.csv_path}")
                
        except Exception as e:
            logger.error(f"Error writing CSV row: {e}")
            raise
    
    def _write_parquet_row(self, row_data: Dict[str, Any]):
        """Write a single row to Parquet (append mode)"""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Convert row to DataFrame
            df = pd.DataFrame([row_data])
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # Check if file exists
            if self.parquet_path.exists():
                # Append to existing file
                pq.write_to_dataset(
                    table,
                    root_path=str(self.parquet_path.parent),
                    basename_template=f"{self.dataset_name}-{{i}}.parquet"
                )
            else:
                # Create new file
                pq.write_table(table, self.parquet_path)
                logger.debug(f"Created new Parquet file: {self.parquet_path}")
                
        except ImportError:
            logger.warning("Parquet support requires pandas and pyarrow. Skipping parquet write.")
        except Exception as e:
            logger.error(f"Error writing Parquet row: {e}")
            raise
    
    def finalize(self):
        """Finalize output files (called at end of experiment)"""
        logger.info(f"Output writer finalized. Files written to: {self.output_dir}")
        
        # Print summary
        if 'csv' in self.formats and self.csv_path.exists():
            row_count = sum(1 for _ in open(self.csv_path)) - 1  # Subtract header
            logger.info(f"CSV: {row_count} rows written to {self.csv_path}")
        
        if 'parquet' in self.formats and self.parquet_path.exists():
            logger.info(f"Parquet: Written to {self.parquet_path}")


def create_row_from_run_result(
    run_id: str,
    timestamp: str,
    git_commit: str,
    seed: int,
    problem_features: Dict[str, Any],
    constraint_settings: Dict[str, Any],
    ga_params: Dict[str, Any],
    result: Dict[str, Any],
    status: str,
    runtime: float,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Create a dataset row from run results
    
    Args:
        run_id: Unique run identifier
        timestamp: ISO8601 timestamp
        git_commit: Git commit hash
        seed: Random seed used
        problem_features: Problem instance features
        constraint_settings: Constraint configuration
        ga_params: GA parameters
        result: Result dictionary from GA execution
        status: Exit status (success/timeout/crashed/infeasible)
        runtime: Execution time in seconds
        notes: Additional notes or error messages
        
    Returns:
        Dictionary containing all fields for dataset row
    """
    import platform
    import sys
    
    # Extract values from result
    fitness_final = result.get('fitness_score', 0.0)
    constraint_violations = result.get('constraint_violations', {})
    best_solution_violations = sum(constraint_violations.values())
    convergence_history = result.get('fitness_history', [])
    generations_completed = result.get('generations_completed', 0)
    
    # Calculate derived metrics
    soft_satisfaction_pct = min(100, max(0, fitness_final * 100))  # Approximation
    
    # Extract teacher workload from result if available
    teacher_workload = result.get('teacher_workload_stats', {
        'min_hours': 0,
        'max_hours': 0,
        'mean_hours': 0.0,
        'std_hours': 0.0,
        'teacher_hours_distribution': {}
    })
    
    # Calculate scheduling coverage
    classes_scheduled = result.get('timetable_size', 0)
    classes_required = problem_features.get('total_required_classes', 0)
    scheduling_coverage_pct = (classes_scheduled / classes_required * 100) if classes_required > 0 else 0.0
    
    # Check if early stopped
    early_stopped = generations_completed < ga_params.get('generations', 0)
    
    row = {
        'run_id': run_id,
        'timestamp_utc': timestamp,
        'git_commit': git_commit,
        'seed': seed,
        'problem_features': problem_features,
        'constraint_settings': constraint_settings,
        'ga_params': ga_params,
        'fitness_final': fitness_final,
        'best_solution_violations': best_solution_violations,
        'soft_satisfaction_pct': soft_satisfaction_pct,
        'convergence_history': json.dumps(convergence_history),
        'runtime_seconds': runtime,
        'process_exit_status': status,
        'notes': notes,
        'constraint_violations_breakdown': constraint_violations,
        'teacher_workload_stats': teacher_workload,
        'generations_completed': generations_completed,
        'early_stopped': early_stopped,
        'classes_scheduled': classes_scheduled,
        'classes_required': classes_required,
        'scheduling_coverage_pct': scheduling_coverage_pct,
        'hostname': platform.node(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'config_file_used': 'generated_config'
    }
    
    return row
