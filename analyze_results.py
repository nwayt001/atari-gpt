"""
Analyze and visualize results from Atari-GPT experiments.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("atari-gpt.analysis")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Atari-GPT experiment results')
    parser.add_argument('--input_dir', default='./experiments',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='./analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to include in analysis (default: all models)')
    parser.add_argument('--games', nargs='+', default=None,
                        help='Games to include in analysis (default: all games)')
    return parser.parse_args()

def find_experiment_dirs(input_dir, models=None, games=None):
    """Find experiment directories matching the specified models and games."""
    experiment_dirs = []
    
    # List all directories in the input directory
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        # Parse the directory name to extract game and model
        parts = item.split('_')
        if len(parts) < 2:
            continue
            
        game = parts[0]
        model = parts[1]
        
        # Filter by model and game if specified
        if models and model not in models:
            continue
        if games and game not in games:
            continue
            
        experiment_dirs.append((item_path, game, model))
    
    return experiment_dirs

def load_results(experiment_dir):
    """Load results from an experiment directory."""
    results = {}
    
    # Try to load actions and rewards
    csv_path = os.path.join(experiment_dir, 'actions_rewards.csv')
    if os.path.exists(csv_path):
        actions = []
        rewards = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        actions.append(int(row[0]))
                        rewards.append(float(row[1]))
            results['actions'] = actions
            results['cumulative_rewards'] = rewards
        except Exception as e:
            logger.error(f"Error loading CSV from {csv_path}: {str(e)}")
    
    return results

def analyze_results(experiment_dirs):
    """Analyze results from all experiment directories."""
    analysis = defaultdict(lambda: defaultdict(dict))
    
    for exp_dir, game, model in experiment_dirs:
        logger.info(f"Analyzing results for {game} with model {model}")
        results = load_results(exp_dir)
        
        if not results:
            logger.warning(f"No results found in {exp_dir}")
            continue
            
        # Calculate metrics
        if 'cumulative_rewards' in results and results['cumulative_rewards']:
            final_reward = results['cumulative_rewards'][-1]
            max_reward = max(results['cumulative_rewards'])
            
            analysis[game][model]['final_reward'] = final_reward
            analysis[game][model]['max_reward'] = max_reward
            analysis[game][model]['rewards'] = results['cumulative_rewards']
            
            logger.info(f"{game} with {model}: Final reward = {final_reward}, Max reward = {max_reward}")
    
    return analysis

def plot_results(analysis, output_dir):
    """Generate plots from the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot final rewards by game and model
    games = list(analysis.keys())
    models = set()
    for game_data in analysis.values():
        models.update(game_data.keys())
    models = list(models)
    
    # Plot reward curves for each game
    for game in games:
        plt.figure(figsize=(12, 6))
        for model in models:
            if model in analysis[game] and 'rewards' in analysis[game][model]:
                rewards = analysis[game][model]['rewards']
                plt.plot(rewards, label=model)
        
        plt.title(f'Cumulative Rewards for {game}')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{game}_rewards.png'))
        plt.close()
    
    # Create a bar chart of final rewards for all games and models
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(games))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        final_rewards = []
        for game in games:
            if model in analysis[game] and 'final_reward' in analysis[game][model]:
                final_rewards.append(analysis[game][model]['final_reward'])
            else:
                final_rewards.append(0)
        
        ax.bar(x + i * width - 0.4 + width/2, final_rewards, width, label=model)
    
    ax.set_xlabel('Games')
    ax.set_ylabel('Final Reward')
    ax.set_title('Final Rewards by Game and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_rewards.png'))
    plt.close()
    
    # Save the analysis as JSON
    with open(os.path.join(output_dir, 'analysis.json'), 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        analysis_dict = {game: {model: data for model, data in game_data.items()} 
                         for game, game_data in analysis.items()}
        
        # Remove the rewards arrays to keep the JSON file small
        for game in analysis_dict:
            for model in analysis_dict[game]:
                if 'rewards' in analysis_dict[game][model]:
                    del analysis_dict[game][model]['rewards']
                    
        json.dump(analysis_dict, f, indent=2)

def main():
    """Main function."""
    args = parse_args()
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(args.input_dir, args.models, args.games)
    
    if not experiment_dirs:
        logger.error(f"No experiment directories found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(experiment_dirs)} experiment directories")
    
    # Analyze results
    analysis = analyze_results(experiment_dirs)
    
    # Plot results
    plot_results(analysis, args.output_dir)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()