#!/usr/bin/env python3
"""
GA Timetable Scheduling Experiment Runner

This script runs the GA across combinations of parameters and problem instances,
capturing structured inputs and outputs for ML training dataset generation.
"""

import argparse
import json
import yaml
import logging
import sys
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import copy
import time

# Add parent directories to path
sys.path.insert(0, '/app/FinalScheduler')
sys.path.insert(0, '/app/experiments/ga_experiments')

from utils import (
    set_seed, get_git_commit, extract_problem_features,
    extract_constraint_violations, extract_teacher_workload_stats,
    run_ga_with_timeout
)
from output_writer import ExperimentOutputWriter, create_row_from_run_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner orchestrating GA executions"""
    
    def __init__(self, config_path: str):
        """
        Initialize experiment runner
        
        Args:
            config_path: Path to experiment configuration YAML
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.git_commit = get_git_commit()
        
        # Extract configuration sections
        self.run_settings = self.config['run_settings']
        self.resource_limits = self.config['resource_limits']
        self.output_config = self.config['output']
        self.ga_param_ranges = self.config['ga_parameters']
        self.problem_config = self.config['problem_instance']
        self.validation_config = self.config.get('validation', {})
        
        # Determine number of workers
        self.num_workers = self._determine_num_workers()
        
        # Initialize output writer
        output_formats = self.output_config.get('formats', ['csv'])
        self.writer = ExperimentOutputWriter(
            output_dir=self.output_config['base_path'],
            dataset_name=self.output_config['dataset_name'],
            formats=output_formats
        )
        
        # Statistics tracking
        self.stats = {
            'total_runs': 0,
            'successful': 0,
            'timeout': 0,
            'crashed': 0,
            'infeasible': 0
        }
        
        logger.info(f"Experiment runner initialized with {self.num_workers} workers")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _determine_num_workers(self) -> int:
        """Determine number of parallel workers"""
        workers_config = self.resource_limits.get('parallel_workers', 'auto')
        
        if workers_config == 'auto':
            cpu_count = mp.cpu_count()
            workers = min(4, max(1, cpu_count - 1))
        else:
            workers = int(workers_config)
        
        return workers
    
    def _generate_ga_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate GA parameter combinations based on sampling mode
        
        Returns:
            List of parameter dictionaries
        """
        sampling_mode = self.run_settings.get('sampling_mode', 'random')
        max_runs = self.run_settings['max_total_runs']
        
        if sampling_mode == 'grid':
            return self._generate_grid_parameters()
        else:
            return self._generate_random_parameters(max_runs)
    
    def _generate_grid_parameters(self) -> List[Dict[str, Any]]:
        """Generate full grid of parameter combinations"""
        param_lists = {}
        
        for param_name, param_config in self.ga_param_ranges.items():
            if param_config.get('type') == 'computed':
                continue  # Skip computed parameters
            
            if param_config['type'] == 'discrete':
                param_lists[param_name] = param_config['values']
            elif param_config['type'] == 'continuous':
                min_val = param_config['min']
                max_val = param_config['max']
                step = param_config.get('step', 0.1)
                values = []
                current = min_val
                while current <= max_val:
                    values.append(round(current, 3))
                    current += step
                param_lists[param_name] = values
        
        # Generate cartesian product
        keys = list(param_lists.keys())
        combinations = list(itertools.product(*[param_lists[k] for k in keys]))
        
        result = []
        for combo in combinations:
            params = dict(zip(keys, combo))
            # Compute derived parameters
            params = self._compute_derived_params(params)
            result.append(params)
        
        logger.info(f"Generated {len(result)} parameter combinations (grid mode)")
        return result
    
    def _generate_random_parameters(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate random parameter samples"""
        seed = self.run_settings.get('random_sample_seed', 42)
        random.seed(seed)
        
        result = []
        
        for _ in range(num_samples):
            params = {}
            
            for param_name, param_config in self.ga_param_ranges.items():
                if param_config.get('type') == 'computed':
                    continue
                
                if param_config['type'] == 'discrete':
                    params[param_name] = random.choice(param_config['values'])
                elif param_config['type'] == 'continuous':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    params[param_name] = random.uniform(min_val, max_val)
            
            # Compute derived parameters
            params = self._compute_derived_params(params)
            result.append(params)
        
        logger.info(f"Generated {len(result)} parameter combinations (random mode)")
        return result
    
    def _compute_derived_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived parameters (e.g., elitism_rate from elite_size)"""
        # Elitism rate
        if 'elite_size' in params and 'population_size' in params:
            params['elitism_rate'] = params['elite_size'] / params['population_size']
        
        # Add any other derived parameters here
        
        return params
    
    def _generate_problem_variations(self) -> List[Dict[str, Any]]:
        """
        Generate problem instance variations
        
        Returns:
            List of problem configuration dictionaries
        """
        base_config_path = self.problem_config['base_config']
        variations_config = self.problem_config['variations']
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
        
        # If no variations, return just the base
        if variations_config.get('use_base_only', False):
            return [base_config]
        
        problem_instances = []
        
        # Generate variations
        if variations_config.get('scale_problem', {}).get('enabled', False):
            scale_config = variations_config['scale_problem']
            for scale_factor in scale_config['scale_factors']:
                scaled = self._scale_problem_instance(base_config, scale_factor, scale_config['scale_components'])
                problem_instances.append(scaled)
        else:
            # Start with base
            problem_instances.append(copy.deepcopy(base_config))
        
        # Apply randomization to each instance
        final_instances = []
        for instance in problem_instances:
            # Randomize teacher availability
            if variations_config.get('randomize_teacher_availability', {}).get('enabled', False):
                availability_config = variations_config['randomize_teacher_availability']
                min_cov = availability_config['coverage_range']['min']
                max_cov = availability_config['coverage_range']['max']
                coverage = random.uniform(min_cov, max_cov)
                instance = self._randomize_teacher_availability(instance, coverage)
            
            # Vary room capacities
            if variations_config.get('vary_room_capacity', {}).get('enabled', False):
                capacity_config = variations_config['vary_room_capacity']
                min_cap = capacity_config['capacity_range']['min']
                max_cap = capacity_config['capacity_range']['max']
                instance = self._vary_room_capacities(instance, min_cap, max_cap)
            
            final_instances.append(instance)
        
        logger.info(f"Generated {len(final_instances)} problem instance variations")
        return final_instances
    
    def _scale_problem_instance(self, config: Dict[str, Any], scale_factor: float, components: List[str]) -> Dict[str, Any]:
        """Scale problem instance by factor"""
        scaled = copy.deepcopy(config)
        
        if 'sections' in components and 'departments' in scaled:
            # Scale sections
            for dept in scaled['departments']:
                if 'sections' in dept:
                    original_count = len(dept['sections'])
                    target_count = max(1, int(original_count * scale_factor))
                    
                    if target_count > original_count:
                        # Duplicate sections
                        while len(dept['sections']) < target_count:
                            for section in dept['sections'][:original_count]:
                                if len(dept['sections']) >= target_count:
                                    break
                                new_section = copy.deepcopy(section)
                                new_section['section_id'] = f"{section['section_id']}_dup{len(dept['sections'])}"
                                dept['sections'].append(new_section)
                    elif target_count < original_count:
                        # Remove sections
                        dept['sections'] = dept['sections'][:target_count]
        
        if 'subjects' in components and 'subjects' in scaled:
            # Scale subjects
            original_count = len(scaled['subjects'])
            target_count = max(1, int(original_count * scale_factor))
            
            if target_count > original_count:
                # Duplicate subjects
                original_subjects = scaled['subjects'][:original_count]
                while len(scaled['subjects']) < target_count:
                    for subj in original_subjects:
                        if len(scaled['subjects']) >= target_count:
                            break
                        new_subj = copy.deepcopy(subj)
                        new_subj['subject_id'] = f"{subj['subject_id']}_dup{len(scaled['subjects'])}"
                        scaled['subjects'].append(new_subj)
            elif target_count < original_count:
                scaled['subjects'] = scaled['subjects'][:target_count]
        
        return scaled
    
    def _randomize_teacher_availability(self, config: Dict[str, Any], coverage: float) -> Dict[str, Any]:
        """Randomize teacher availability with given coverage"""
        modified = copy.deepcopy(config)
        
        if 'faculty' in modified and 'time_slots' in modified:
            periods = modified['time_slots'].get('periods', [])
            working_days = len(modified['time_slots'].get('working_days', []))
            
            for faculty in modified['faculty']:
                # Determine how many slots to make unavailable
                total_slots = len(periods) * working_days
                unavailable_count = int(total_slots * (1 - coverage))
                
                # Randomly select slots
                unavailable = []
                for _ in range(unavailable_count):
                    day = random.randint(0, working_days - 1)
                    period_idx = random.randint(0, len(periods) - 1)
                    period_id = periods[period_idx]['id']
                    unavailable.append({'day': day, 'period': period_id, 'reason': 'random'})
                
                faculty['unavailable_periods'] = unavailable
        
        return modified
    
    def _vary_room_capacities(self, config: Dict[str, Any], min_cap: int, max_cap: int) -> Dict[str, Any]:
        """Randomize room capacities within range"""
        modified = copy.deepcopy(config)
        
        if 'rooms' in modified:
            for room in modified['rooms']:
                room['capacity'] = random.randint(min_cap, max_cap)
        
        return modified
    
    def _execute_single_run(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single GA run with given configuration
        
        Args:
            run_config: Dictionary containing all run parameters
            
        Returns:
            Dictionary with run results
        """
        run_id = run_config['run_id']
        seed = run_config['seed']
        ga_params = run_config['ga_params']
        problem_config = run_config['problem_config']
        timeout = run_config['timeout']
        
        logger.info(f"Starting run {run_id} with seed {seed}")
        
        try:
            # Merge GA params into problem config
            full_config = copy.deepcopy(problem_config)
            full_config['genetic_algorithm_params'] = ga_params
            
            # Run GA with timeout
            status, result, runtime = run_ga_with_timeout(full_config, seed, timeout)
            
            # Extract problem features
            from timetable_generator import TimetableData
            data = TimetableData(config_dict=full_config)
            problem_features = extract_problem_features(data)
            
            # Create output row
            timestamp = datetime.utcnow().isoformat() + 'Z'
            
            row = create_row_from_run_result(
                run_id=run_id,
                timestamp=timestamp,
                git_commit=self.git_commit,
                seed=seed,
                problem_features=problem_features,
                constraint_settings=full_config.get('constraints', {}),
                ga_params=ga_params,
                result=result,
                status=status,
                runtime=runtime,
                notes=result.get('error', '') if status != 'success' else ''
            )
            
            # Full solution encoding for raw JSON
            raw_solution = {
                'run_id': run_id,
                'config': full_config,
                'result': result,
                'status': status,
                'runtime': runtime
            }
            
            return {
                'status': status,
                'row': row,
                'raw_solution': raw_solution,
                'config_used': full_config
            }
            
        except Exception as e:
            logger.error(f"Run {run_id} failed with exception: {e}")
            return {
                'status': 'crashed',
                'row': None,
                'error': str(e)
            }
    
    def run_experiments(self):
        """Main execution method - run all experiments"""
        logger.info("Starting experiment execution")
        
        # Generate parameter combinations
        ga_param_combos = self._generate_ga_parameter_combinations()
        
        # Generate problem variations
        problem_variations = self._generate_problem_variations()
        
        # Create all run configurations
        run_configs = []
        repetitions = self.run_settings.get('repetitions_per_config', 1)
        timeout = self.resource_limits['per_run_timeout_seconds']
        
        for ga_params in ga_param_combos:
            for problem_config in problem_variations:
                for rep in range(repetitions):
                    run_id = str(uuid.uuid4())
                    seed = random.randint(0, 2**31 - 1)
                    
                    run_configs.append({
                        'run_id': run_id,
                        'seed': seed,
                        'ga_params': ga_params,
                        'problem_config': problem_config,
                        'timeout': timeout,
                        'repetition': rep
                    })
        
        # Limit to max_total_runs
        max_runs = self.run_settings['max_total_runs']
        if len(run_configs) > max_runs:
            logger.warning(f"Generated {len(run_configs)} configs, limiting to {max_runs}")
            run_configs = random.sample(run_configs, max_runs)
        
        logger.info(f"Total runs to execute: {len(run_configs)}")
        
        # Execute runs in parallel
        completed = 0
        failed_count = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._execute_single_run, config): config 
                      for config in run_configs}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    status = result['status']
                    
                    # Update statistics
                    self.stats['total_runs'] += 1
                    self.stats[status] = self.stats.get(status, 0) + 1
                    
                    if status == 'success' and result.get('row'):
                        # Write to output
                        self.writer.write_row(
                            row_data=result['row'],
                            raw_solution=result.get('raw_solution'),
                            config_used=result.get('config_used')
                        )
                        completed += 1
                    else:
                        failed_count += 1
                    
                    # Log progress
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(run_configs)} completed, {failed_count} failed")
                    
                    # Check failure rate
                    if self.stats['total_runs'] > 20:
                        failure_rate = failed_count / self.stats['total_runs']
                        max_failure_rate = self.validation_config.get('max_failure_rate', 0.5)
                        
                        if failure_rate > max_failure_rate:
                            logger.error(f"Failure rate {failure_rate:.1%} exceeds threshold {max_failure_rate:.1%}")
                            logger.error("Halting experiment execution")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                    
                except Exception as e:
                    logger.error(f"Error processing future: {e}")
                    failed_count += 1
        
        # Finalize output
        self.writer.finalize()
        
        # Print final statistics
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total runs: {self.stats['total_runs']}")
        logger.info(f"Successful: {self.stats.get('success', 0)}")
        logger.info(f"Timeout: {self.stats.get('timeout', 0)}")
        logger.info(f"Crashed: {self.stats.get('crashed', 0)}")
        logger.info(f"Infeasible: {self.stats.get('infeasible', 0)}")
        logger.info("=" * 80)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Run GA timetable scheduling experiments for ML dataset generation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML file'
    )
    
    parser.add_argument(
        '--max-runs',
        type=int,
        default=None,
        help='Override max_total_runs from config'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Override number of parallel workers'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner(args.config)
    
    # Apply overrides
    if args.max_runs:
        runner.run_settings['max_total_runs'] = args.max_runs
        logger.info(f"Override: max_total_runs = {args.max_runs}")
    
    if args.workers:
        runner.num_workers = args.workers
        logger.info(f"Override: num_workers = {args.workers}")
    
    if args.dry_run:
        logger.info("DRY RUN - Configuration loaded successfully")
        logger.info(f"Config: {args.config}")
        logger.info(f"Max runs: {runner.run_settings['max_total_runs']}")
        logger.info(f"Workers: {runner.num_workers}")
        logger.info(f"Sampling mode: {runner.run_settings['sampling_mode']}")
        return
    
    # Run experiments
    try:
        runner.run_experiments()
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
