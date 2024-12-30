import torch
import argparse
from utils import set_seed, get_device, Logger


def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, default=42, help='Seed value')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    logger = Logger()
    device = get_device()