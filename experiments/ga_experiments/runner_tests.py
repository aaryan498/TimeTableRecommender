#!/usr/bin/env python3
"""
Unit tests for experiment runner

Tests runner logic without running expensive GA operations
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, '/app/experiments/ga_experiments')

from output_writer import ExperimentOutputWriter, create_row_from_run_result
from utils import get_git_commit, set_seed


class TestOutputWriter(unittest.TestCase):
    """Test output writer functionality"""
    
    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_writer_initialization(self):
        """Test writer initializes correctly"""
        writer = ExperimentOutputWriter(
            output_dir=self.temp_dir,
            dataset_name="test_dataset",
            formats=['csv']
        )
        
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue((Path(self.temp_dir) / 'raw').exists())
        self.assertTrue((Path(self.temp_dir) / 'configs').exists())
    
    def test_csv_write(self):
        """Test CSV writing"""
        writer = ExperimentOutputWriter(
            output_dir=self.temp_dir,
            dataset_name="test_dataset",
            formats=['csv']
        )
        
        # Create sample row
        row_data = {
            'run_id': 'test-123',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'seed': 42,
            'fitness_final': 0.85,
            'runtime_seconds': 10.5,
            'process_exit_status': 'success'
        }
        
        writer.write_row(row_data)
        
        # Check CSV file exists
        csv_path = Path(self.temp_dir) / 'test_dataset.csv'
        self.assertTrue(csv_path.exists())
        
        # Check content
        with open(csv_path, 'r') as f:
            content = f.read()
            self.assertIn('run_id', content)
            self.assertIn('test-123', content)
    
    def test_multiple_rows(self):
        """Test writing multiple rows"""
        writer = ExperimentOutputWriter(
            output_dir=self.temp_dir,
            dataset_name="test_dataset",
            formats=['csv']
        )
        
        # Write multiple rows
        for i in range(5):
            row_data = {
                'run_id': f'test-{i}',
                'timestamp_utc': '2025-01-01T00:00:00Z',
                'seed': i,
                'fitness_final': 0.8 + i * 0.01,
                'runtime_seconds': 10.0 + i,
                'process_exit_status': 'success'
            }
            writer.write_row(row_data)
        
        # Count rows
        csv_path = Path(self.temp_dir) / 'test_dataset.csv'
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            # Should have header + 5 data rows
            self.assertEqual(len(lines), 6)
    
    def test_raw_json_output(self):
        """Test raw JSON solution output"""
        writer = ExperimentOutputWriter(
            output_dir=self.temp_dir,
            dataset_name="test_dataset",
            formats=['csv']
        )
        
        row_data = {
            'run_id': 'test-json',
            'fitness_final': 0.9
        }
        
        raw_solution = {
            'timetable': [{'section': 'A', 'subject': 'Math'}],
            'fitness': 0.9
        }
        
        writer.write_row(row_data, raw_solution=raw_solution)
        
        # Check JSON file exists
        json_path = Path(self.temp_dir) / 'raw' / 'test-json.json'
        self.assertTrue(json_path.exists())
        
        # Validate JSON content
        with open(json_path, 'r') as f:
            loaded = json.load(f)
            self.assertEqual(loaded['fitness'], 0.9)


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility"""
        import random
        import numpy as np
        
        set_seed(42)
        val1 = random.random()
        arr1 = np.random.rand(5)
        
        set_seed(42)
        val2 = random.random()
        arr2 = np.random.rand(5)
        
        self.assertEqual(val1, val2)
        self.assertTrue(np.array_equal(arr1, arr2))
    
    def test_git_commit(self):
        """Test git commit retrieval"""
        commit = get_git_commit()
        self.assertIsInstance(commit, str)
        self.assertTrue(len(commit) > 0)


class TestRowCreation(unittest.TestCase):
    """Test dataset row creation"""
    
    def test_create_row_success(self):
        """Test creating row from successful run"""
        problem_features = {
            'num_courses': 10,
            'num_teachers': 5,
            'total_required_classes': 50
        }
        
        ga_params = {
            'population_size': 50,
            'generations': 100,
            'mutation_rate': 0.2
        }
        
        result = {
            'fitness_score': 0.85,
            'constraint_violations': {'clash': 2},
            'fitness_history': [0.5, 0.7, 0.85],
            'generations_completed': 100,
            'timetable_size': 45
        }
        
        row = create_row_from_run_result(
            run_id='test-123',
            timestamp='2025-01-01T00:00:00Z',
            git_commit='abc123',
            seed=42,
            problem_features=problem_features,
            constraint_settings={},
            ga_params=ga_params,
            result=result,
            status='success',
            runtime=120.5
        )
        
        # Validate row structure
        self.assertEqual(row['run_id'], 'test-123')
        self.assertEqual(row['seed'], 42)
        self.assertEqual(row['fitness_final'], 0.85)
        self.assertEqual(row['process_exit_status'], 'success')
        self.assertEqual(row['runtime_seconds'], 120.5)
        self.assertEqual(row['generations_completed'], 100)
    
    def test_create_row_timeout(self):
        """Test creating row from timeout"""
        row = create_row_from_run_result(
            run_id='test-timeout',
            timestamp='2025-01-01T00:00:00Z',
            git_commit='abc123',
            seed=42,
            problem_features={},
            constraint_settings={},
            ga_params={'generations': 200},
            result={'error': 'timeout'},
            status='timeout',
            runtime=300.0,
            notes='Execution timed out after 300s'
        )
        
        self.assertEqual(row['process_exit_status'], 'timeout')
        self.assertEqual(row['runtime_seconds'], 300.0)
        self.assertIn('timed out', row['notes'].lower())


class TestSchemaValidation(unittest.TestCase):
    """Test schema validation"""
    
    def test_schema_exists(self):
        """Test that schema.json exists and is valid"""
        schema_path = Path('/app/experiments/ga_experiments/schema.json')
        self.assertTrue(schema_path.exists())
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Validate schema structure
        self.assertIn('fields', schema)
        self.assertIsInstance(schema['fields'], list)
        self.assertTrue(len(schema['fields']) > 0)
        
        # Check required fields exist
        field_names = [f['name'] for f in schema['fields']]
        required_fields = [
            'run_id', 'timestamp_utc', 'seed', 'fitness_final',
            'runtime_seconds', 'process_exit_status'
        ]
        
        for field in required_fields:
            self.assertIn(field, field_names)


class TestConfigParsing(unittest.TestCase):
    """Test configuration parsing"""
    
    def test_sample_config_exists(self):
        """Test that sample config exists and is valid YAML"""
        config_path = Path('/app/experiments/ga_experiments/config/samples/experiment_config.yaml')
        self.assertTrue(config_path.exists())
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate config structure
        self.assertIn('run_settings', config)
        self.assertIn('ga_parameters', config)
        self.assertIn('problem_instance', config)
        self.assertIn('resource_limits', config)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
