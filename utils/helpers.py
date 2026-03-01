import os
import random
import numpy as np
import torch
import cv2
from PIL import Image

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        config.CHECKPOINT_DIR,
        config.RESULTS_DIR,
        config.LOGS_DIR,
        os.path.join(config.TRAIN_DIR, 'images'),
        os.path.join(config.TRAIN_DIR, 'masks'),
        os.path.join(config.VAL_DIR, 'images'),
        os.path.join(config.VAL_DIR, 'masks'),
        os.path.join(config.TEST_DIR, 'images'),
        os.path.join(config.TEST_DIR, 'masks'),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ All directories created successfully")

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
