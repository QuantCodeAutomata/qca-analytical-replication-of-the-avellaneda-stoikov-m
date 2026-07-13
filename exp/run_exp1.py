"""Experiment 1: Analytical verification of Avellaneda-Stoikov formulas."""
import sys
sys.path.insert(0, '/workspace/project')

from src.avellaneda_stoikov.experiments import run_analytical_experiment

if __name__ == '__main__':
    run_analytical_experiment()
