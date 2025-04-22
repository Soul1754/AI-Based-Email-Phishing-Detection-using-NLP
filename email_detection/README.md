# AI-based Email Detection using BERT and NLP

## Overview

A hybrid BERT-LSTM model for email classification combining contextual understanding and sequence modeling. Features include:

- Spam/phishing detection
- Model explainability using LIME
- Training metrics visualization
- Batch processing capabilities

## Features

- **Hybrid Architecture**: BERT embeddings + Bidirectional LSTM
- **Explainable AI**: Integrated LIME explanations
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Training Visualization**: Loss/accuracy curves

## Installation

```bash
git clone https://github.com/yourusername/email_detection.git
cd email_detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: All scripts must be run with Python 3. Use `python3` command explicitly for both installation and running the model.

## Configuration

Create `.env` file with:

```ini
# Data paths
DATA_PATH=./data/phishing_email.csv

# Model paths
MODEL_DIR=./models
BERT_MODEL_PATH=./models/bert_email_classifier.pt

# Output paths
EXPLANATIONS_DIR=./models/explanations
PLOTS_DIR=./plots

# Model parameters
MAX_LENGTH=128
BATCH_SIZE=16
NUM_EPOCHS=5
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
DROPOUT_RATE=0.1
NUM_EXPLANATIONS=5
```

## Training Workflow

1. Prepare dataset in `data/` directory
2. Start training:

```bash
python src/train.py \
  --batch_size 16 \
  --num_epochs 5 \
  --learning_rate 2e-5
```

3. View training metrics in `plots/training_history.png`

## Inference

Load trained model for predictions:

```python
from model import BertLSTMClassifier

model = BertLSTMClassifier.load_model('models/bert_email_classifier.pt')
prediction = model.classify_email("Your email text here")
```

## Model Architecture

```python
class BertLSTMClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size*2, 2)
```

## Explainability

Generate LIME explanations for predictions:

```bash
python src/explain.py --input "Your email text here"
```

Outputs saved in `models/explanations/`

## Web Interface (Future Work)

```bash
# Planned integration
python src/app.py
```

Access at `http://localhost:5000`

## Project Structure

```
email_detection/
├── data/              # Dataset storage
├── models/            # Saved models & explanations
├── plots/             # Training metrics visualizations
└── src/
    ├── model.py       # BERT-LSTM architecture
    └── train.py       # Training pipeline
```
