import json
import argparse
import logging
import os
import sys
from run_experiments import Run

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

# Suppress httpx and urllib3 logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("atari-gpt")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Atari-GPT evaluations')
    parser.add_argument('--provider', nargs='+', default=['rand', 'openai', 'gemini', 'claude'],
                        help='Models to evaluate (default: all providers)')
    parser.add_argument('--games', nargs='+', default=None,
                        help='Specific games to evaluate (default: all games)')
    parser.add_argument('--model_name', default=None,
                        help='Specific model name you would like to test (default: gpt-4.1-mini)')
    parser.add_argument('--output_dir', default='./experiments',
                        help='Directory to save experiment results')
    args = parser.parse_args()
    
    # If 'all' is specified, replace it with the list of all models
    if 'all' in args.provider:
        args.provider = ['rand', 'openai', 'gemini', 'claude']

    model_id = args.model_name
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment configurations
    try:
        with open('envs.json') as file:
            environment_list = json.load(file)
    except FileNotFoundError:
        logger.error("envs.json file not found. Please make sure it exists in the current directory.")
    except json.JSONDecodeError:
        logger.error("Error parsing envs.json. Please check the file format.")
    
    # Filter games if specified
    if args.games:
        environments = {k: v for k, v in environment_list.items() if k in args.games}
        if not environments:
            logger.error(f"None of the specified games {args.games} found in envs.json")
    else:
        environments = environment_list
    
    # Run evaluations for each model and game
    for provider in args.provider:
        logger.info(f"Starting evaluation for model: {provider}")
        for game, prompt in environments.items():
            try:
                logger.info(f"Running test for: {game}")
                Run(game, provider, args.output_dir, prompt, model_id)
                logger.info(f"Completed test for: {game}")
            except Exception as e:
                if model_id == None:
                    model_id = 'gpt-4.1-mini'
                logger.error(f"Error running test for {game} with model {model_id} from {provider}: {str(e)}")
                continue
        
        logger.info(f"Completed evaluation for model: {provider}")

if __name__ == "__main__":
    main()
    
