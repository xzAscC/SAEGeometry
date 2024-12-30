import torch
import argparse
from utils import set_seed, get_device
from logger import setup_logger


def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed value')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger()
    device = get_device()
    logger.info(f"Using device {device}")
    logger.info(f'all arguments: {args}')
    logger.info(f"step 1: Load model, data and sae to {device}")
    logger.info(f"step 2: get the activation of the SAE")
    logger.info(f"step 3: Geometry analysis")
    logger.info(f"step 3.1: vectors' cos sim in the same layer(max and min)")
    # TODO: pairwise in the same layer, too large to plot
    logger.info(f"step 3.2: cos sim with unembedding matrix")

    # TODO: here we do not care about the meaning, we only care about the cos sim and freq
    logger.info(f"step 4: Frequency analysis")
    logger.info(f"step 4.1: Plot avg frequency of the activation of the SAE")
    logger.info(f"step 4.2: cos sim with high freq, low freq and between them")
    logger.info(f"step 4.3: freq of the high cos sim, low cos sim")
    logger.info(f"step 4.4: freq of the high cos sim, low cos sim between the unembedding matrix")

    logger.info(f"Then we can save the results and see the ablation study")