"""
Utility functions for metrics computation
"""

import torch
import numpy as np


def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        accuracy: Classification accuracy
    """
    pred_labels = torch.argmax(predictions, dim=1)
    correct = (pred_labels == targets).sum().item()
    total = targets.size(0)
    return correct / total


def compute_precision_recall_f1(predictions, targets, num_classes):
    """
    Compute precision, recall, and F1 score
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        precision, recall, f1: Metrics for each class
    """
    pred_labels = torch.argmax(predictions, dim=1)
    
    precision = []
    recall = []
    f1 = []
    
    for c in range(num_classes):
        true_positive = ((pred_labels == c) & (targets == c)).sum().item()
        false_positive = ((pred_labels == c) & (targets != c)).sum().item()
        false_negative = ((pred_labels != c) & (targets == c)).sum().item()
        
        prec = true_positive / (true_positive + false_positive + 1e-10)
        rec = true_positive / (true_positive + false_negative + 1e-10)
        f1_score = 2 * (prec * rec) / (prec + rec + 1e-10)
        
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    
    return precision, recall, f1
