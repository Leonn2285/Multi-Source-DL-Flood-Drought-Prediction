"""
Configuration file for Big Data and Deep Learning-Based Natural Disaster Prediction
Using Multi-Source Environmental Data
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ========================== PATHS ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "SEA_2024_FINAL_CLEAN.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ========================== FEATURE GROUPS ==========================
# Satellite Features (CNN Encoder) - Spatial patterns
SATELLITE_FEATURES = ["ndvi", "evi", "lst"]

# Weather/Meteorological Features (LSTM Encoder) - Temporal sequences
WEATHER_FEATURES = [
    "precip_mm",      # Precipitation
    "temp_c",         # Temperature
    "dewpoint_c",     # Dewpoint temperature
    "wind_u",         # Wind U component
    "wind_v",         # Wind V component
    "evap_mm",        # Evaporation
    "pressure_hpa",   # Atmospheric pressure
    "soil_temp_c"     # Soil temperature
]

# Static Features (MLP Encoder) - Geographic context
STATIC_FEATURES = ["elevation", "landcover", "lat", "lon"]

# All features combined
ALL_FEATURES = SATELLITE_FEATURES + WEATHER_FEATURES + STATIC_FEATURES

# ========================== DATA CONFIG ==========================
@dataclass
class DataConfig:
    """Data configuration settings"""
    # Split ratios
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Sequence length for LSTM (days lookback)
    sequence_length: int = 7
    
    # Grid size for CNN spatial input (e.g., 5x5 neighboring cells)
    grid_size: int = 5
    
    # Batch size
    batch_size: int = 256
    
    # Number of workers for data loading (0 for macOS to avoid multiprocessing issues)
    num_workers: int = 0
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Handle missing values
    fill_missing_strategy: str = "median"  # Options: "median", "mean", "zero", "forward_fill"

# ========================== MODEL CONFIG ==========================
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder output dimension (each encoder outputs this dimension)
    encoder_output_dim: int = 128
    
    # CNN Encoder for Satellite Features
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.3
    
    # LSTM Encoder for Weather Features
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True
    
    # MLP Encoder for Static Features
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [64, 128])
    mlp_dropout: float = 0.3
    
    # Fusion Layer
    fusion_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    fusion_dropout: float = 0.4
    
    # Prediction Heads
    # Flood: Binary classification (0: No flood, 1: Flood)
    flood_num_classes: int = 2
    
    # Drought: Multi-class classification (0: None, 1: Mild, 2: Moderate, 3: Severe)
    # Based on data, currently binary (0, 1), can extend later
    drought_num_classes: int = 2

# ========================== TRAINING CONFIG ==========================
@dataclass
class TrainingConfig:
    """Training configuration settings"""
    # Number of epochs
    num_epochs: int = 100
    
    # Learning rate
    learning_rate: float = 1e-3
    
    # Weight decay for regularization
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    
    # Class weights for imbalanced data (will be computed from data)
    use_class_weights: bool = True
    
    # Multi-task learning loss weights
    flood_loss_weight: float = 1.0
    drought_loss_weight: float = 1.0
    
    # Gradient clipping
    gradient_clip_value: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Checkpoint settings
    save_best_only: bool = True
    checkpoint_metric: str = "val_f1_avg"  # Monitor average F1 score

# ========================== EXPERIMENT CONFIG ==========================
@dataclass
class ExperimentConfig:
    """Experiment configuration combining all configs"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment name
    experiment_name: str = "disaster_prediction_v1"
    
    # Device
    device: str = "cuda"  # Will be updated based on availability
    
    def __post_init__(self):
        import torch
        if not torch.cuda.is_available():
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"

# ========================== DEFAULT CONFIG ==========================
def get_config(fast_mode: bool = False) -> ExperimentConfig:
    """
    Get experiment configuration
    
    Args:
        fast_mode: If True, uses smaller model and fewer epochs for faster training
    """
    config = ExperimentConfig()
    
    if fast_mode:
        # ===== FAST MODE SETTINGS =====
        # Reduce epochs significantly
        config.training.num_epochs = 30
        
        # Increase batch size for faster training
        config.data.batch_size = 512
        
        # Reduce early stopping patience
        config.training.early_stopping_patience = 5
        config.training.scheduler_patience = 3
        
        # Smaller model for faster training
        config.model.encoder_output_dim = 64  # 128 → 64
        config.model.cnn_channels = [16, 32, 64]  # Smaller CNN
        config.model.lstm_hidden_size = 64  # 128 → 64
        config.model.lstm_num_layers = 1  # 2 → 1 layer
        config.model.mlp_hidden_sizes = [32, 64]  # Smaller MLP
        config.model.fusion_hidden_sizes = [128, 64]  # Smaller fusion
        
        # Update experiment name
        config.experiment_name = "disaster_prediction_fast"
        
    return config


def get_fast_config() -> ExperimentConfig:
    """Shortcut to get fast training configuration"""
    return get_config(fast_mode=True)

if __name__ == "__main__":
    # Print configuration for verification
    config = get_config()
    print(f"Device: {config.device}")
    print(f"Satellite Features: {SATELLITE_FEATURES}")
    print(f"Weather Features: {WEATHER_FEATURES}")
    print(f"Static Features: {STATIC_FEATURES}")
    print(f"Encoder output dim: {config.model.encoder_output_dim}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
