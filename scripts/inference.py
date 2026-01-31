"""
Inference script for DGFormer
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DGFormer


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with DGFormer')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output.txt',
                        help='Path to save output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create model (with default parameters - adjust as needed)
    model = DGFormer().to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # TODO: Load input data and run inference
    print(f'Running inference on {args.input}')
    # Add your inference logic here
    
    print(f'Results will be saved to {args.output}')


if __name__ == '__main__':
    main()
