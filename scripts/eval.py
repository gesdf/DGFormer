"""
Evaluation script for DGFormer
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DGFormer
from datasets import SpatialRelationDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DGFormer')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create model
    model = DGFormer(
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers']
    ).to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # TODO: Load test dataset and evaluate
    print('Evaluation script is set up. Add your test data to begin evaluation.')


if __name__ == '__main__':
    main()
