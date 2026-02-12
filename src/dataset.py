"""
Dataset classes for Natural Disaster Prediction
Handles data loading, preprocessing, and creating sequences for multi-encoder architecture
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    SATELLITE_FEATURES, WEATHER_FEATURES, STATIC_FEATURES,
    DataConfig, DATA_PATH, MODEL_DIR
)


class DisasterDataProcessor:
    """
    Preprocesses raw data for the disaster prediction model.
    Handles missing values, normalization, and creates train/val/test splits.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, Dict] = {}
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load raw data from CSV"""
        filepath = filepath or DATA_PATH
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        df = df.copy()
        strategy = self.config.fill_missing_strategy
        
        numeric_features = SATELLITE_FEATURES + WEATHER_FEATURES + STATIC_FEATURES
        
        for col in numeric_features:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if fit:
                    if strategy == "median":
                        fill_value = df[col].median()
                    elif strategy == "mean":
                        fill_value = df[col].mean()
                    elif strategy == "zero":
                        fill_value = 0
                    else:
                        fill_value = df[col].median()
                    
                    self.feature_stats[col] = {"fill_value": fill_value}
                else:
                    fill_value = self.feature_stats.get(col, {}).get("fill_value", 0)
                
                df[col] = df[col].fillna(fill_value)
                print(f"Filled {missing_count:,} missing values in '{col}' with {fill_value:.4f}")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize numerical features using StandardScaler"""
        df = df.copy()
        
        feature_groups = {
            "satellite": SATELLITE_FEATURES,
            "weather": WEATHER_FEATURES,
            "static": [f for f in STATIC_FEATURES if f not in ["landcover"]]  # landcover is categorical
        }
        
        for group_name, features in feature_groups.items():
            valid_features = [f for f in features if f in df.columns]
            
            if fit:
                scaler = StandardScaler()
                df[valid_features] = scaler.fit_transform(df[valid_features])
                self.scalers[group_name] = scaler
            else:
                if group_name in self.scalers:
                    df[valid_features] = self.scalers[group_name].transform(df[valid_features])
        
        # Landcover: keep as numeric (it's already integer encoded)
        # Just normalize it like other features instead of using LabelEncoder
        if "landcover" in df.columns:
            if fit:
                scaler = StandardScaler()
                df["landcover"] = scaler.fit_transform(df[["landcover"]].astype(float))
                self.scalers["landcover"] = scaler
            else:
                if "landcover" in self.scalers:
                    df["landcover"] = self.scalers["landcover"].transform(df[["landcover"]].astype(float))
        
        return df
    
    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits, stratified by grid_id to prevent data leakage"""
        # Get unique grid_ids
        unique_grids = df["grid_id"].unique()
        
        # Split grid_ids
        train_grids, temp_grids = train_test_split(
            unique_grids,
            train_size=self.config.train_ratio,
            random_state=self.config.random_seed
        )
        
        val_ratio_adjusted = self.config.valid_ratio / (self.config.valid_ratio + self.config.test_ratio)
        val_grids, test_grids = train_test_split(
            temp_grids,
            train_size=val_ratio_adjusted,
            random_state=self.config.random_seed
        )
        
        # Create dataframes for each split
        train_df = df[df["grid_id"].isin(train_grids)].copy()
        val_df = df[df["grid_id"].isin(val_grids)].copy()
        test_df = df[df["grid_id"].isin(test_grids)].copy()
        
        print(f"\nData splits:")
        print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Valid: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def process(self, filepath: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Full preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)
        
        # Parse date and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)
        
        # Create splits first (to prevent data leakage)
        train_df, val_df, test_df = self.create_splits(df)
        
        # Fit preprocessing on train, apply to all
        train_df = self.handle_missing_values(train_df, fit=True)
        val_df = self.handle_missing_values(val_df, fit=False)
        test_df = self.handle_missing_values(test_df, fit=False)
        
        train_df = self.normalize_features(train_df, fit=True)
        val_df = self.normalize_features(val_df, fit=False)
        test_df = self.normalize_features(test_df, fit=False)
        
        return train_df, val_df, test_df
    
    def save_preprocessors(self, filepath: str = None):
        """Save scalers and encoders for inference"""
        filepath = filepath or os.path.join(MODEL_DIR, "preprocessors.pkl")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessors = {
            "scalers": self.scalers,
            "label_encoders": self.label_encoders,
            "feature_stats": self.feature_stats,
            "config": self.config
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(preprocessors, f)
        print(f"Saved preprocessors to {filepath}")
    
    def load_preprocessors(self, filepath: str = None):
        """Load preprocessors for inference"""
        filepath = filepath or os.path.join(MODEL_DIR, "preprocessors.pkl")
        
        with open(filepath, "rb") as f:
            preprocessors = pickle.load(f)
        
        self.scalers = preprocessors["scalers"]
        self.label_encoders = preprocessors["label_encoders"]
        self.feature_stats = preprocessors["feature_stats"]
        self.config = preprocessors["config"]
        print(f"Loaded preprocessors from {filepath}")


class DisasterDataset(Dataset):
    """
    PyTorch Dataset for disaster prediction.
    Creates sequences for LSTM and spatial patches for CNN.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 7,
        grid_size: int = 5,
        satellite_features: List[str] = None,
        weather_features: List[str] = None,
        static_features: List[str] = None
    ):
        self.df = df.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.grid_size = grid_size
        
        self.satellite_features = satellite_features or SATELLITE_FEATURES
        self.weather_features = weather_features or WEATHER_FEATURES
        self.static_features = static_features or STATIC_FEATURES
        
        # Build grid index for efficient lookup
        self._build_grid_index()
        
        # Create valid samples (only those with enough history)
        self._create_valid_samples()
        
    def _build_grid_index(self):
        """Build index for efficient grid cell lookup"""
        # Create unique (lat, lon) grid mapping
        self.df["lat_idx"] = pd.factorize(self.df["lat"])[0]
        self.df["lon_idx"] = pd.factorize(self.df["lon"])[0]
        
        # Store unique grid coordinates
        self.unique_lats = np.sort(self.df["lat"].unique())
        self.unique_lons = np.sort(self.df["lon"].unique())
        
        # Create lookup for grid_id -> indices
        self.grid_date_index = self.df.groupby(["grid_id", "date"]).indices
        
    def _create_valid_samples(self):
        """Create list of valid (grid_id, date) pairs with sufficient history"""
        self.valid_samples = []
        
        grouped = self.df.groupby("grid_id")
        
        for grid_id, group in grouped:
            dates = group["date"].sort_values().unique()
            
            # Start from sequence_length to ensure enough history
            for i in range(self.sequence_length - 1, len(dates)):
                target_date = dates[i]
                # Check if we have continuous dates
                start_date = dates[max(0, i - self.sequence_length + 1)]
                
                self.valid_samples.append({
                    "grid_id": grid_id,
                    "target_date": target_date,
                    "target_idx": group[group["date"] == target_date].index[0]
                })
        
        print(f"Created {len(self.valid_samples):,} valid samples")
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def _get_satellite_features(self, idx: int) -> torch.Tensor:
        """
        Get satellite features for CNN encoder.
        For now, we use point-based features; in production, you would load actual satellite imagery.
        Shape: (channels, height, width) - simulated as (n_features, grid_size, grid_size)
        """
        row = self.df.iloc[idx]
        
        # Get satellite features as a vector
        features = row[self.satellite_features].values.astype(np.float32)
        
        # Simulate spatial grid by creating a simple feature map
        # In production, replace with actual satellite imagery patches
        n_features = len(self.satellite_features)
        
        # Create a pseudo-spatial representation (center value with slight variations)
        feature_map = np.zeros((n_features, self.grid_size, self.grid_size), dtype=np.float32)
        center = self.grid_size // 2
        
        for i, val in enumerate(features):
            # Simple approach: center has actual value, surroundings have slight noise
            feature_map[i, :, :] = val
            # Add some spatial variation (Gaussian-like decay from center)
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    dist = np.sqrt((y - center)**2 + (x - center)**2)
                    feature_map[i, y, x] *= (1 - 0.1 * dist / center) if center > 0 else 1
        
        return torch.from_numpy(feature_map)
    
    def _get_weather_sequence(self, grid_id: str, target_date: pd.Timestamp) -> torch.Tensor:
        """
        Get weather feature sequence for LSTM encoder.
        Shape: (sequence_length, n_features)
        """
        # Get all rows for this grid sorted by date
        grid_data = self.df[self.df["grid_id"] == grid_id].sort_values("date")
        
        # Find the target date position
        target_mask = grid_data["date"] == target_date
        target_pos = grid_data[target_mask].index[0]
        
        # Get the sequence up to and including target date
        grid_data_before = grid_data[grid_data["date"] <= target_date].tail(self.sequence_length)
        
        # Extract weather features
        sequence = grid_data_before[self.weather_features].values.astype(np.float32)
        
        # Pad if necessary (not enough history)
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), len(self.weather_features)), dtype=np.float32)
            sequence = np.vstack([padding, sequence])
        
        return torch.from_numpy(sequence)
    
    def _get_static_features(self, idx: int) -> torch.Tensor:
        """
        Get static features for MLP encoder.
        Shape: (n_features,)
        """
        row = self.df.iloc[idx]
        features = row[self.static_features].values.astype(np.float32)
        return torch.from_numpy(features)
    
    def _get_labels(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get flood and drought labels"""
        row = self.df.iloc[idx]
        flood = torch.tensor(row["flood"], dtype=torch.long)
        drought = torch.tensor(row["drought"], dtype=torch.long)
        return flood, drought
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.valid_samples[idx]
        target_idx = sample["target_idx"]
        grid_id = sample["grid_id"]
        target_date = sample["target_date"]
        
        # Get features from different encoders
        satellite_features = self._get_satellite_features(target_idx)
        weather_sequence = self._get_weather_sequence(grid_id, target_date)
        static_features = self._get_static_features(target_idx)
        
        # Get labels
        flood_label, drought_label = self._get_labels(target_idx)
        
        return {
            "satellite": satellite_features,      # (C, H, W) for CNN
            "weather": weather_sequence,          # (seq_len, features) for LSTM
            "static": static_features,            # (features,) for MLP
            "flood_label": flood_label,           # scalar
            "drought_label": drought_label,       # scalar
            "grid_id": grid_id,
            "date": str(target_date)
        }


def collate_fn(batch):
    """Custom collate function to handle string fields"""
    return {
        "satellite": torch.stack([x["satellite"] for x in batch]),
        "weather": torch.stack([x["weather"] for x in batch]),
        "static": torch.stack([x["static"] for x in batch]),
        "flood_label": torch.stack([x["flood_label"] for x in batch]),
        "drought_label": torch.stack([x["drought_label"] for x in batch]),
        "grid_id": [x["grid_id"] for x in batch],
        "date": [x["date"] for x in batch]
    }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: DataConfig = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test sets"""
    config = config or DataConfig()
    
    # Create datasets
    train_dataset = DisasterDataset(
        train_df,
        sequence_length=config.sequence_length,
        grid_size=config.grid_size
    )
    
    val_dataset = DisasterDataset(
        val_df,
        sequence_length=config.sequence_length,
        grid_size=config.grid_size
    )
    
    test_dataset = DisasterDataset(
        test_df,
        sequence_length=config.sequence_length,
        grid_size=config.grid_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Compute class weights for imbalanced classes"""
    from sklearn.utils.class_weight import compute_class_weight
    
    weights = {}
    
    # Flood weights
    flood_classes = np.unique(df["flood"])
    flood_weights = compute_class_weight("balanced", classes=flood_classes, y=df["flood"])
    weights["flood"] = torch.tensor(flood_weights, dtype=torch.float32)
    
    # Drought weights  
    drought_classes = np.unique(df["drought"])
    drought_weights = compute_class_weight("balanced", classes=drought_classes, y=df["drought"])
    weights["drought"] = torch.tensor(drought_weights, dtype=torch.float32)
    
    print(f"Class weights computed:")
    print(f"  Flood: {weights['flood'].tolist()}")
    print(f"  Drought: {weights['drought'].tolist()}")
    
    return weights


if __name__ == "__main__":
    # Test data processing
    processor = DisasterDataProcessor()
    train_df, val_df, test_df = processor.process()
    
    # Save preprocessors
    processor.save_preprocessors()
    
    # Compute class weights
    weights = compute_class_weights(train_df)
    
    # Create dataloaders
    config = DataConfig()
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, config)
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Satellite: {batch['satellite'].shape}")
    print(f"  Weather: {batch['weather'].shape}")
    print(f"  Static: {batch['static'].shape}")
    print(f"  Flood labels: {batch['flood_label'].shape}")
    print(f"  Drought labels: {batch['drought_label'].shape}")
