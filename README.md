# 🌊🏜️ Natural Disaster Prediction with Deep Learning

## Big Data and Deep Learning-Based Natural Disaster Prediction Using Multi-Source Environmental Data

This project implements a multi-encoder deep learning model for predicting **floods** and **droughts** using multi-source environmental data from Southeast Asia (2024).

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATASET                                │
│           (1 row = 1 grid cell × 1 day, ~1.3M+ rows)           │
└─────────────────────────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   CNN Encoder   │  │  LSTM Encoder   │  │   MLP Encoder   │
    │  (128-dim out)  │  │  (128-dim out)  │  │  (128-dim out)  │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
    │ Satellite:      │  │ Weather:        │  │ Static:         │
    │ - NDVI          │  │ - precip_mm     │  │ - elevation     │
    │ - EVI           │  │ - temp_c        │  │ - landcover     │
    │ - LST           │  │ - dewpoint_c    │  │ - lat, lon      │
    │                 │  │ - wind_u/v      │  │                 │
    │                 │  │ - evap_mm       │  │                 │
    │                 │  │ - pressure_hpa  │  │                 │
    │                 │  │ - soil_temp_c   │  │                 │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Mid-Level Fusion      │
                    │   (Concatenate + MLP)   │
                    └─────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌─────────────────┐         ┌─────────────────┐
          │  Flood Head     │         │  Drought Head   │
          │  (Binary)       │         │  (Binary)       │
          │  0: No Flood    │         │  0: No Drought  │
          │  1: Flood       │         │  1: Drought     │
          └─────────────────┘         └─────────────────┘
```

---

## 📁 Project Structure

```
DL for disaster/
├── SEA_2024_FINAL_CLEAN.csv      # Preprocessed dataset (~1.3M rows)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── configs/
│   ├── __init__.py
│   └── config.py                  # All configurations (data, model, training)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Data loading and preprocessing
│   ├── models.py                  # CNN, LSTM, MLP encoders + Fusion
│   └── utils.py                   # Metrics, logging, checkpointing
│
├── notebooks/
│   ├── train.ipynb                # Training notebook (run this first)
│   └── evaluate.ipynb             # Evaluation and visualization
│
├── models/                        # Saved model checkpoints
│   └── (trained models will be saved here)
│
└── logs/                          # Training logs and plots
    └── (logs will be saved here)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Open and run `notebooks/train.ipynb`:
- Loads and preprocesses the data
- Creates train/val/test splits (70/15/15)
- Trains the multi-encoder model
- Saves best checkpoint based on validation F1

### 3. Evaluate the Model

Open and run `notebooks/evaluate.ipynb`:
- Loads the best trained model
- Generates predictions on test set
- Computes comprehensive metrics
- Creates visualizations (confusion matrices, ROC curves, etc.)

---

## 📊 Data Description

| Feature Group | Features | Encoder |
|---------------|----------|---------|
| **Satellite** | NDVI, EVI, LST | CNN |
| **Weather** | precip_mm, temp_c, dewpoint_c, wind_u, wind_v, evap_mm, pressure_hpa, soil_temp_c | LSTM |
| **Static** | elevation, landcover, lat, lon | MLP |

### Labels
- **Flood**: Binary (0 = No flood, 1 = Flood)
- **Drought**: Binary (0 = No drought, 1 = Drought)

### Data Stats
- **Total samples**: ~1.3 million
- **Flood positive rate**: ~5%
- **Drought positive rate**: ~2%

---

## ⚙️ Configuration

All configurations are in `configs/config.py`:

```python
# Data settings
sequence_length = 7      # Days of history for LSTM
grid_size = 5           # Spatial grid for CNN
batch_size = 256

# Model settings
encoder_output_dim = 128
lstm_hidden_size = 128
lstm_num_layers = 2

# Training settings
num_epochs = 100
learning_rate = 1e-3
early_stopping_patience = 10
```

---

## 🏗️ Model Details

### CNN Encoder (Satellite Features)
- Processes spatial patterns from vegetation indices and land surface temperature
- 3 convolutional layers with batch normalization
- Global average pooling → 128-dim output

### LSTM Encoder (Weather Features)
- Processes 7-day weather sequences
- Bidirectional LSTM with attention mechanism
- Captures temporal dependencies → 128-dim output

### MLP Encoder (Static Features)
- Processes geographic context (elevation, landcover, coordinates)
- 2-layer MLP with dropout → 128-dim output

### Mid-Level Fusion
- Concatenates all encoder outputs (128 × 3 = 384 dims)
- MLP fusion layers → 128-dim fused representation

### Prediction Heads
- **Flood Head**: 128 → 64 → 2 (binary classification)
- **Drought Head**: 128 → 64 → 2 (binary classification)

---

## 📈 Training Features

- **Class Weighting**: Handles imbalanced dataset
- **Mixed Precision**: FP16 training on GPU (faster + less memory)
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stabilizes training
- **Checkpointing**: Saves best model automatically

---

## 📊 Evaluation Metrics

The model is evaluated on:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve

---

## 🖥️ Hardware Requirements

- **Minimum**: 16GB RAM, CPU only (slow training)
- **Recommended**: 32GB RAM, GPU with 8GB+ VRAM
- **Apple Silicon**: MPS acceleration supported

---

## 📚 References

- Dataset: Southeast Asia Environmental Data 2024
- Framework: PyTorch
- Multi-task Learning for disaster prediction

---

## 📝 License

This project is for educational and research purposes.

---

## 👥 Authors

Natural Disaster Prediction Research Team
# Multi-Source-DL-Flood-Drought-Prediction
