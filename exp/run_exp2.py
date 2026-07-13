"""Experiment 2: Monte Carlo replication of inventory strategy vs symmetric benchmark."""
import sys
sys.path.insert(0, '/workspace/project')

from src.avellaneda_stoikov.experiments import run_monte_carlo_experiment

if __name__ == '__main__':
    run_monte_carlo_experiment()
