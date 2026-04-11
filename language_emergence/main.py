import argparse
import json
import torch
import os

from configs.experiment_configs import ExperimentConfig
# TODO: implement these modules
from src.rl.teacher import generate_q_matrices
from src.models.autoencoder import train_language_model
from src.evaluation.student_eval import run_evaluations
from src.utils.plotting import generate_all_plots

def main():
    parser = argparse.ArgumentParser(description="Language Emergence Execution Pipeline")
    parser.add_argument('--stage', type=str, required=True, 
                        choices=['all', 'generate_teacher', 'train_language', 'evaluate', 'plot'],
                        help='Which stage of the pipeline to run')
    
    args = parser.parse_args()
    config = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config['data_dir'], exist_ok=True)

    # STAGE 1: Q-Matrix Generation (Teacher)
    if args.stage in ['all', 'generate_teacher']:
        print("--- Starting Teacher Q-Matrix Generation ---")
        # from cell 17
        q_matrices, label_dict, wall_state_dict = generate_q_matrices(config, device)
        # Save these to config['data_dir']

    # STAGE 2: Autoencoder Training (Language Generation)
    if args.stage in ['all', 'train_language']:
        print("--- Starting Language Training (Autoencoder) ---")
        # from cell 18
        model, losses = train_language_model(config, device)
        # Save model weights to config['data_dir']

    # STAGE 3: Student Performance Evaluation
    if args.stage in ['all', 'evaluate']:
        print("--- Starting Student Evaluation ---")
        # from cell 19
        run_evaluations(config, device)
        # Saves solving rates to text files

    # STAGE 4: Visualization
    if args.stage in ['all', 'plot']:
        print("--- Generating Plots ---")
        # from cells 20-29
        generate_all_plots(config)

if __name__ == "__main__":
    main()