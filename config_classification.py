import os

class Config:
    # Paths
    DATA_DIR = 'data/raw/diagnosis-of-diabetic-retinopathy/Diagnosis of Diabetic Retinopathy'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'valid')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    
    CHECKPOINT_DIR = 'checkpoints'
    RESULTS_DIR = 'results'
    LOGS_DIR = 'logs'
    
    # Model parameters
    IMAGE_SIZE = 224  # Standard for classification
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    
    # Model architecture
    MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'efficientnet_b0', 'vgg16'
    PRETRAINED = True
    
    # Number of classes
    NUM_CLASSES = 2  # DR vs No_DR
    CLASS_NAMES = ['No_DR', 'DR']
    
    # Training parameters
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # Device
    DEVICE = 'cuda'  # Will auto-detect in code
    
    # Random seed
    SEED = 42
