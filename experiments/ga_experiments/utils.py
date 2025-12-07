"""
Utility functions for GA experiment runner
Handles safe invocation, seeding, metrics extraction, and more
"""

import json
import random
import numpy as np
import copy
import subprocess
import sys
import time
import os
import signal
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # Note: The GA code has its own SEED setting, we'll override it


def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd='/app',
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def extract_problem_features(data_obj) -> Dict[str, Any]:
    """
    Extract problem instance features from TimetableData object
    
    Args:
        data_obj: TimetableData instance
        
    Returns:
        Dictionary of problem features
    """
    try:
        # Count various elements
        num_courses = len(data_obj.subjects)
        num_labs = len(data_obj.labs)
        num_rooms = len(data_obj.rooms)
        num_teachers = len(data_obj.faculty)
        num_sections = len(data_obj.sections)
        num_departments = len(data_obj.departments)
        total_periods = len(data_obj.period_ids)
        working_days = data_obj.num_working_days
        
        # Room capacity statistics
        capacities = [r.get('capacity', 0) for r in data_obj.rooms.values() if r.get('capacity')]
        room_capacity_min = min(capacities) if capacities else 0
        room_capacity_max = max(capacities) if capacities else 0
        room_capacity_mean = np.mean(capacities) if capacities else 0
        room_capacity_std = np.std(capacities) if capacities else 0
        
        # Teacher availability coverage
        total_slots = working_days * total_periods
        unavailable_count = 0
        for fac in data_obj.faculty.values():
            unavailable_count += len(fac.get('unavailable_periods', []))
        
        teacher_availability = 1.0 - (unavailable_count / (num_teachers * total_slots)) if num_teachers > 0 else 1.0
        
        # Conflicting course pairs (approximation)
        # In this codebase, conflicts are implicit through scheduling constraints
        conflicting_pairs = 0
        
        # Total required classes
        total_required = 0
        for section_id, section in data_obj.sections.items():
            for subj in data_obj.subjects.values():
                if _subject_applies_to_section(subj, section, data_obj):
                    total_required += subj.get('lectures_per_week', 1)
            for lab in data_obj.labs.values():
                if _subject_applies_to_section(lab, section, data_obj):
                    total_required += lab.get('sessions_per_week', 1)
        
        avg_classes_per_section = total_required / num_sections if num_sections > 0 else 0
        
        # Average subjects per teacher
        total_subjects_assigned = sum(len(subjects) for subjects in data_obj.faculty_subjects.values())
        avg_subjects_per_teacher = total_subjects_assigned / num_teachers if num_teachers > 0 else 0
        
        return {
            'num_courses': num_courses,
            'num_labs': num_labs,
            'num_rooms': num_rooms,
            'num_teachers': num_teachers,
            'num_sections': num_sections,
            'num_departments': num_departments,
            'total_periods': total_periods,
            'working_days': working_days,
            'room_capacity_min': int(room_capacity_min),
            'room_capacity_mean': float(room_capacity_mean),
            'room_capacity_max': int(room_capacity_max),
            'room_capacity_std': float(room_capacity_std),
            'teacher_availability_coverage': float(teacher_availability),
            'conflicting_course_pairs': conflicting_pairs,
            'total_required_classes': total_required,
            'avg_classes_per_section': float(avg_classes_per_section),
            'avg_subjects_per_teacher': float(avg_subjects_per_teacher)
        }
    except Exception as e:
        logger.error(f"Error extracting problem features: {e}")
        return {}


def _subject_applies_to_section(subject: Dict, section: Dict, data_obj) -> bool:
    """Check if a subject applies to a section"""
    subject_depts = subject.get('departments', [])
    if subject_depts:
        section_dept = data_obj.section_department.get(section['section_id'])
        if section_dept and section_dept not in subject_depts:
            return False
    return True


def extract_constraint_violations(chromosome) -> Dict[str, int]:
    """
    Extract detailed constraint violation breakdown from chromosome
    
    Args:
        chromosome: TimetableChromosome instance
        
    Returns:
        Dictionary mapping constraint names to violation counts
    """
    try:
        violations = {}
        
        # The chromosome has constraint_violations dict
        if hasattr(chromosome, 'constraint_violations'):
            violations = dict(chromosome.constraint_violations)
        
        # Also check fitness_breakdown if available
        if hasattr(chromosome, 'fitness_breakdown'):
            for key, value in chromosome.fitness_breakdown.items():
                if 'violation' in key.lower() or 'clash' in key.lower():
                    violations[key] = value
        
        return violations
    except Exception as e:
        logger.error(f"Error extracting constraint violations: {e}")
        return {}


def extract_teacher_workload_stats(chromosome) -> Dict[str, Any]:
    """
    Extract per-teacher workload statistics
    
    Args:
        chromosome: TimetableChromosome instance
        
    Returns:
        Dictionary with workload statistics
    """
    try:
        # Count hours per teacher
        teacher_hours = {}
        for entry in chromosome.timetable:
            if entry.faculty_id:
                teacher_hours[entry.faculty_id] = teacher_hours.get(entry.faculty_id, 0) + 1
        
        if not teacher_hours:
            return {
                'min_hours': 0,
                'max_hours': 0,
                'mean_hours': 0.0,
                'std_hours': 0.0,
                'teacher_hours_distribution': {}
            }
        
        hours_list = list(teacher_hours.values())
        
        return {
            'min_hours': int(min(hours_list)),
            'max_hours': int(max(hours_list)),
            'mean_hours': float(np.mean(hours_list)),
            'std_hours': float(np.std(hours_list)),
            'teacher_hours_distribution': teacher_hours
        }
    except Exception as e:
        logger.error(f"Error extracting teacher workload: {e}")
        return {}


def extract_convergence_history(ga_instance) -> List[float]:
    """
    Extract convergence history (best fitness per generation)
    
    Args:
        ga_instance: GeneticAlgorithm instance
        
    Returns:
        List of fitness scores
    """
    try:
        if hasattr(ga_instance, 'fitness_history'):
            return ga_instance.fitness_history
        return []
    except Exception:
        return []


def serialize_solution_encoding(chromosome) -> Dict[str, Any]:
    """
    Serialize full solution encoding for JSON storage
    
    Args:
        chromosome: TimetableChromosome instance
        
    Returns:
        Dictionary containing full solution details
    """
    try:
        encoding = {
            'timetable_entries': [],
            'fitness_score': chromosome.fitness_score,
            'constraint_violations': dict(chromosome.constraint_violations),
            'fitness_breakdown': getattr(chromosome, 'fitness_breakdown', {})
        }
        
        # Serialize each timetable entry
        for entry in chromosome.timetable:
            encoding['timetable_entries'].append({
                'section_id': entry.section_id,
                'subject_id': entry.subject_id,
                'faculty_id': entry.faculty_id,
                'room_id': entry.room_id,
                'day': entry.time_slot.day,
                'period': entry.time_slot.period,
                'entry_type': entry.entry_type,
                'batch': getattr(entry, 'batch', ''),
                'lab_session_id': getattr(entry, 'lab_session_id', ''),
                'is_lab_second_period': getattr(entry, 'is_lab_second_period', False)
            })
        
        return encoding
    except Exception as e:
        logger.error(f"Error serializing solution: {e}")
        return {}


def calculate_soft_satisfaction_pct(chromosome) -> float:
    """
    Calculate percentage of soft constraints satisfied
    
    Args:
        chromosome: TimetableChromosome instance
        
    Returns:
        Percentage (0-100)
    """
    try:
        # This is an approximation - soft constraints are embedded in fitness
        # A higher fitness generally means better soft constraint satisfaction
        
        # If fitness is normalized 0-1, we can use it directly
        if 0 <= chromosome.fitness_score <= 1:
            return chromosome.fitness_score * 100
        
        # Otherwise, use a heuristic
        # Check if fitness_breakdown has soft constraint components
        if hasattr(chromosome, 'fitness_breakdown'):
            soft_components = ['balanced_daily_load', 'faculty_preference', 'gap_penalty']
            soft_score = sum(chromosome.fitness_breakdown.get(key, 0) for key in soft_components)
            # Normalize to percentage
            return min(100, max(0, soft_score))
        
        # Default: estimate from overall fitness
        return min(100, max(0, chromosome.fitness_score * 100))
    except Exception:
        return 0.0


def run_ga_with_timeout(config_dict: Dict[str, Any], seed: int, timeout: int) -> Tuple[str, Dict[str, Any], float]:
    """
    Run GA with timeout using subprocess isolation
    
    Args:
        config_dict: Configuration dictionary
        seed: Random seed
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (status, result_dict, runtime)
        status: "success", "timeout", "crashed", "infeasible"
    """
    start_time = time.time()
    
    try:
        # Create a temporary script that runs the GA
        import tempfile
        import pickle
        
        # Save config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            config_file = f.name
        
        # Create runner script
        runner_script = f"""
import sys
import json
import random
import numpy as np
import pickle
import traceback

sys.path.insert(0, '/app/FinalScheduler')

from timetable_generator import TimetableData, TimetableChromosome, GeneticAlgorithm

def run():
    try:
        # Set seeds
        seed = {seed}
        random.seed(seed)
        np.random.seed(seed)
        
        # Load config
        config_file = "{config_file}"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Override GA params in config
        if 'genetic_algorithm_params' not in config:
            config['genetic_algorithm_params'] = {{}}
        
        # Create TimetableData
        data = TimetableData(config_dict=config)
        
        # Create and run GA
        ga = GeneticAlgorithm(data)
        ga.initialize_population()
        ga.evolve()
        
        # Get best solution
        best = ga.get_best_solution()
        # Handle case where get_best_solution returns a list
        if isinstance(best, list) and best:
            best = best[0]
        
        # Extract results
        result = {{
            'fitness_score': best.fitness_score,
            'constraint_violations': dict(best.constraint_violations),
            'fitness_breakdown': getattr(best, 'fitness_breakdown', {{}}),
            'timetable_size': len(best.timetable),
            'fitness_history': getattr(ga, 'fitness_history', []),
            'generations_completed': getattr(ga, 'generation', 0)
        }}
        
        # Serialize solution
        solution = {{}}
        for entry in best.timetable:
            solution[len(solution)] = {{
                'section_id': entry.section_id,
                'subject_id': entry.subject_id,
                'faculty_id': entry.faculty_id,
                'room_id': entry.room_id,
                'day': entry.time_slot.day,
                'period': entry.time_slot.period
            }}
        result['solution'] = solution
        
        # Write result
        with open('{config_file}.result', 'w') as f:
            json.dump(result, f)
        
        sys.exit(0)
        
    except Exception as e:
        error = {{
            'error': str(e),
            'traceback': traceback.format_exc()
        }}
        with open('{config_file}.result', 'w') as f:
            json.dump(error, f)
        sys.exit(1)

if __name__ == '__main__':
    run()
"""
        
        # Write runner script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(runner_script)
            script_file = f.name
        
        # Run with timeout
        try:
            result = subprocess.run(
                [sys.executable, script_file],
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            runtime = time.time() - start_time
            
            # Read result
            result_file = f"{config_file}.result"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    output = json.load(f)
                
                # Clean up temp files
                os.unlink(config_file)
                os.unlink(script_file)
                os.unlink(result_file)
                
                if 'error' in output:
                    return 'crashed', {'error': output['error'], 'traceback': output.get('traceback')}, runtime
                
                return 'success', output, runtime
            else:
                os.unlink(config_file)
                os.unlink(script_file)
                return 'crashed', {'error': 'No result file created'}, runtime
                
        except subprocess.TimeoutExpired:
            runtime = time.time() - start_time
            # Clean up
            try:
                os.unlink(config_file)
                os.unlink(script_file)
                if os.path.exists(f"{config_file}.result"):
                    os.unlink(f"{config_file}.result")
            except:
                pass
            return 'timeout', {'error': 'Execution timeout'}, runtime
            
    except Exception as e:
        runtime = time.time() - start_time
        logger.error(f"Error in run_ga_with_timeout: {e}")
        return 'crashed', {'error': str(e)}, runtime
