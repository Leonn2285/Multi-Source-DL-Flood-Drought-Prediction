"""
Deep Learning Models for Natural Disaster Prediction
Architecture: Multi-encoder (CNN + LSTM + MLP) with Mid-Level Fusion

CNN Encoder: Processes satellite/spatial features (NDVI, EVI, LST)
LSTM Encoder: Processes temporal weather sequences
MLP Encoder: Processes static geographic features
Fusion Module: Combines encoder outputs
Prediction Heads: Flood (binary) and Drought (multi-class) classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ModelConfig, SATELLITE_FEATURES, WEATHER_FEATURES, STATIC_FEATURES


class CNNEncoder(nn.Module):
    """
    CNN Encoder for Satellite/Spatial Features
    Processes 2D spatial patterns from satellite imagery (NDVI, EVI, LST)
    
    Input: (batch, channels, height, width) - e.g., (B, 3, 5, 5)
    Output: (batch, encoder_output_dim) - e.g., (B, 128)
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # NDVI, EVI, LST
        channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        output_dim: int = 128,
        dropout: float = 0.3,
        grid_size: int = 5
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # Build convolutional layers
        layers = []
        prev_channels = in_channels
        
        for i, out_channels in enumerate(channels):
            layers.append(
                nn.Conv2d(
                    prev_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            # Add pooling every other layer if grid is large enough
            if i < len(channels) - 1 and grid_size > 3:
                layers.append(nn.MaxPool2d(2, stride=1))  # Slight reduction
            
            prev_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final projection to output dimension
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Satellite features (B, C, H, W)
        Returns:
            Feature vector (B, output_dim)
        """
        # Convolutional feature extraction
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Project to output dimension
        x = self.fc(x)
        
        return x


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder for Temporal Weather Features
    Processes sequential weather data over time
    
    Input: (batch, seq_len, features) - e.g., (B, 7, 8)
    Output: (batch, encoder_output_dim) - e.g., (B, 128)
    """
    
    def __init__(
        self,
        input_size: int = 8,  # Number of weather features
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism for weighted combination of time steps
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Weather sequence (B, seq_len, features)
        Returns:
            Feature vector (B, output_dim)
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (B, seq_len, hidden_size * num_directions)
        
        # Attention-weighted combination
        attn_weights = self.attention(lstm_out)  # (B, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden_size * num_directions)
        
        # Project to output dimension
        output = self.fc(context)
        
        return output


class MLPEncoder(nn.Module):
    """
    MLP Encoder for Static Geographic Features
    Processes elevation, landcover, lat, lon
    
    Input: (batch, features) - e.g., (B, 4)
    Output: (batch, encoder_output_dim) - e.g., (B, 128)
    """
    
    def __init__(
        self,
        input_size: int = 4,  # elevation, landcover, lat, lon
        hidden_sizes: List[int] = [64, 128],
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Static features (B, features)
        Returns:
            Feature vector (B, output_dim)
        """
        return self.mlp(x)


class MidLevelFusion(nn.Module):
    """
    Mid-Level Fusion Module
    Combines encoded features from CNN, LSTM, and MLP encoders
    
    Input: Three feature vectors of shape (B, encoder_dim)
    Output: Fused representation (B, output_dim)
    """
    
    def __init__(
        self,
        encoder_dim: int = 128,
        hidden_sizes: List[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.4,
        fusion_type: str = "concat"  # Options: "concat", "attention", "gated"
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        # Total input size after concatenation (3 encoders)
        total_input = encoder_dim * 3
        
        if fusion_type == "attention":
            # Cross-attention fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=encoder_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            total_input = encoder_dim * 3
            
        elif fusion_type == "gated":
            # Gated fusion with learnable weights
            self.gate_cnn = nn.Sequential(
                nn.Linear(encoder_dim * 3, encoder_dim),
                nn.Sigmoid()
            )
            self.gate_lstm = nn.Sequential(
                nn.Linear(encoder_dim * 3, encoder_dim),
                nn.Sigmoid()
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(encoder_dim * 3, encoder_dim),
                nn.Sigmoid()
            )
        
        # Fusion MLP
        layers = []
        prev_size = total_input
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.fusion_mlp = nn.Sequential(*layers)
        
    def forward(
        self,
        cnn_features: torch.Tensor,
        lstm_features: torch.Tensor,
        mlp_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cnn_features: CNN encoder output (B, encoder_dim)
            lstm_features: LSTM encoder output (B, encoder_dim)
            mlp_features: MLP encoder output (B, encoder_dim)
        Returns:
            Fused features (B, output_dim)
        """
        if self.fusion_type == "attention":
            # Stack features for attention
            stacked = torch.stack([cnn_features, lstm_features, mlp_features], dim=1)  # (B, 3, dim)
            attn_out, _ = self.attention(stacked, stacked, stacked)
            fused = attn_out.view(attn_out.size(0), -1)
            
        elif self.fusion_type == "gated":
            # Concatenate for gate computation
            concat = torch.cat([cnn_features, lstm_features, mlp_features], dim=1)
            
            # Compute gates
            g_cnn = self.gate_cnn(concat)
            g_lstm = self.gate_lstm(concat)
            g_mlp = self.gate_mlp(concat)
            
            # Apply gates
            gated_cnn = g_cnn * cnn_features
            gated_lstm = g_lstm * lstm_features
            gated_mlp = g_mlp * mlp_features
            
            fused = torch.cat([gated_cnn, gated_lstm, gated_mlp], dim=1)
            
        else:  # concat (default)
            fused = torch.cat([cnn_features, lstm_features, mlp_features], dim=1)
        
        return self.fusion_mlp(fused)


class PredictionHead(nn.Module):
    """
    Prediction Head for classification tasks
    Can be used for both flood (binary) and drought (multi-class) prediction
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused features (B, input_dim)
        Returns:
            Logits (B, num_classes)
        """
        return self.head(x)


class DisasterPredictionModel(nn.Module):
    """
    Complete Multi-Encoder Disaster Prediction Model
    
    Architecture:
    1. CNN Encoder → Satellite features (NDVI, EVI, LST)
    2. LSTM Encoder → Weather sequence (temp, precipitation, wind, etc.)
    3. MLP Encoder → Static features (elevation, landcover, coordinates)
    4. Mid-Level Fusion → Combine all encoder outputs
    5. Prediction Heads → Flood (binary) & Drought (multi-class)
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # ===== ENCODERS =====
        # CNN Encoder for satellite features
        self.cnn_encoder = CNNEncoder(
            in_channels=len(SATELLITE_FEATURES),
            channels=self.config.cnn_channels,
            kernel_size=self.config.cnn_kernel_size,
            output_dim=self.config.encoder_output_dim,
            dropout=self.config.cnn_dropout
        )
        
        # LSTM Encoder for weather sequences
        self.lstm_encoder = LSTMEncoder(
            input_size=len(WEATHER_FEATURES),
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            output_dim=self.config.encoder_output_dim,
            dropout=self.config.lstm_dropout,
            bidirectional=self.config.lstm_bidirectional
        )
        
        # MLP Encoder for static features
        self.mlp_encoder = MLPEncoder(
            input_size=len(STATIC_FEATURES),
            hidden_sizes=self.config.mlp_hidden_sizes,
            output_dim=self.config.encoder_output_dim,
            dropout=self.config.mlp_dropout
        )
        
        # ===== FUSION =====
        self.fusion = MidLevelFusion(
            encoder_dim=self.config.encoder_output_dim,
            hidden_sizes=self.config.fusion_hidden_sizes,
            output_dim=self.config.encoder_output_dim,
            dropout=self.config.fusion_dropout,
            fusion_type="concat"  # Can change to "attention" or "gated"
        )
        
        # ===== PREDICTION HEADS =====
        # Flood prediction (binary classification)
        self.flood_head = PredictionHead(
            input_dim=self.config.encoder_output_dim,
            num_classes=self.config.flood_num_classes
        )
        
        # Drought prediction (multi-class classification)
        self.drought_head = PredictionHead(
            input_dim=self.config.encoder_output_dim,
            num_classes=self.config.drought_num_classes
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        static: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            satellite: Satellite features (B, C, H, W)
            weather: Weather sequence (B, seq_len, features)
            static: Static features (B, features)
            
        Returns:
            Dictionary containing:
                - flood_logits: Flood prediction logits (B, 2)
                - drought_logits: Drought prediction logits (B, num_classes)
                - cnn_features: CNN encoder output (B, encoder_dim)
                - lstm_features: LSTM encoder output (B, encoder_dim)
                - mlp_features: MLP encoder output (B, encoder_dim)
                - fused_features: Fused representation (B, encoder_dim)
        """
        # Encode each modality
        cnn_features = self.cnn_encoder(satellite)
        lstm_features = self.lstm_encoder(weather)
        mlp_features = self.mlp_encoder(static)
        
        # Fuse encoded features
        fused_features = self.fusion(cnn_features, lstm_features, mlp_features)
        
        # Predictions
        flood_logits = self.flood_head(fused_features)
        drought_logits = self.drought_head(fused_features)
        
        return {
            "flood_logits": flood_logits,
            "drought_logits": drought_logits,
            "cnn_features": cnn_features,
            "lstm_features": lstm_features,
            "mlp_features": mlp_features,
            "fused_features": fused_features
        }
    
    def predict(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        static: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions (with probabilities)
        
        Returns:
            Dictionary with predicted classes and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(satellite, weather, static)
            
            flood_probs = F.softmax(outputs["flood_logits"], dim=1)
            drought_probs = F.softmax(outputs["drought_logits"], dim=1)
            
            return {
                "flood_pred": torch.argmax(flood_probs, dim=1),
                "flood_prob": flood_probs[:, 1],  # Probability of flood
                "drought_pred": torch.argmax(drought_probs, dim=1),
                "drought_prob": drought_probs
            }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters in each component"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            "cnn_encoder": count_params(self.cnn_encoder),
            "lstm_encoder": count_params(self.lstm_encoder),
            "mlp_encoder": count_params(self.mlp_encoder),
            "fusion": count_params(self.fusion),
            "flood_head": count_params(self.flood_head),
            "drought_head": count_params(self.drought_head),
            "total": count_params(self)
        }


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining flood and drought prediction losses
    Supports class weighting for imbalanced datasets
    """
    
    def __init__(
        self,
        flood_weight: float = 1.0,
        drought_weight: float = 1.0,
        flood_class_weights: torch.Tensor = None,
        drought_class_weights: torch.Tensor = None
    ):
        super().__init__()
        
        self.flood_weight = flood_weight
        self.drought_weight = drought_weight
        
        # Loss functions with optional class weights
        self.flood_loss_fn = nn.CrossEntropyLoss(weight=flood_class_weights)
        self.drought_loss_fn = nn.CrossEntropyLoss(weight=drought_class_weights)
        
    def forward(
        self,
        flood_logits: torch.Tensor,
        drought_logits: torch.Tensor,
        flood_labels: torch.Tensor,
        drought_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Returns:
            Dictionary with individual and total losses
        """
        flood_loss = self.flood_loss_fn(flood_logits, flood_labels)
        drought_loss = self.drought_loss_fn(drought_logits, drought_labels)
        
        total_loss = self.flood_weight * flood_loss + self.drought_weight * drought_loss
        
        return {
            "flood_loss": flood_loss,
            "drought_loss": drought_loss,
            "total_loss": total_loss
        }


def create_model(config: ModelConfig = None, device: str = "cpu") -> DisasterPredictionModel:
    """Factory function to create and initialize the model"""
    config = config or ModelConfig()
    model = DisasterPredictionModel(config)
    model = model.to(device)
    
    # Print model summary
    params = model.count_parameters()
    print(f"\n{'='*50}")
    print(f"Model Parameter Summary:")
    print(f"{'='*50}")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    print(f"{'='*50}\n")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    import torch
    
    config = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = create_model(config, device)
    
    # Create dummy inputs
    batch_size = 4
    satellite = torch.randn(batch_size, 3, 5, 5).to(device)  # (B, C, H, W)
    weather = torch.randn(batch_size, 7, 8).to(device)  # (B, seq_len, features)
    static = torch.randn(batch_size, 4).to(device)  # (B, features)
    
    # Forward pass
    outputs = model(satellite, weather, static)
    
    print("Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test predictions
    predictions = model.predict(satellite, weather, static)
    print("\nPredictions:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
