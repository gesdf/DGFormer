"""
Training script for DGFormer
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DGFormer
from datasets import SpatialRelationDataset
from utils.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train DGFormer')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
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
    
    # Create datasets and dataloaders
    # TODO: Implement actual dataset loading
    print('Loading datasets...')
    # train_dataset = SpatialRelationDataset(config['data']['train_dir'])
    # val_dataset = SpatialRelationDataset(config['data']['val_dir'])
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['training']['batch_size'],
    #     shuffle=True,
    #     num_workers=config['data']['num_workers']
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config['training']['batch_size'],
    #     shuffle=False,
    #     num_workers=config['data']['num_workers']
    # )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # TODO: Create trainer and start training
    print('Starting training...')
    # trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device, config)
    # trainer.train()
    
    print('Training script is set up. Add your training data to begin training.')


if __name__ == '__main__':
    main()
