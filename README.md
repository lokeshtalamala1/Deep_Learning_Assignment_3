# Deep_Learning_Assignment_3
Use recurrent neural networks to build a transliteration system.
-------------------------------------------------------------------

# DA6401 - Assignment 3: Sequence-to-Sequence Transliteration

This project implements sequence-to-sequence models (vanilla and attention-based) for transliteration using RNN, GRU, and LSTM architectures. It supports training, evaluation, hyperparameter optimization, and visualization, with integration for Weights & Biases (W&B).

---

## Features

- Encoder-decoder models with RNN, GRU, or LSTM cells
- Attention mechanism support
- Character-level tokenization
- Configurable hyperparameters via command line or config file
- Training, evaluation, and error analysis
- Visualizations (heatmaps, confusion matrices, etc.)
- W&B integration for experiment tracking and sweeps

---

## Dataset

- Uses the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) by Google Research.
- Download and extract the dataset to `dakshina_dataset_v1.0` in your project root, or update the data path in `config.py`.

---

## Project Structure

```
.
├── dakshina_dataset_v1.0/     # Dataset directory
├── predictions/
│   ├── predictions_attention/ # Attention model predictions
│   └── predictions_vanilla/   # Vanilla model predictions
├── config.py                  # Configuration management
├── dataloader.py              # Data loading and preprocessing
├── evaluate.py                # Model evaluation and visualization
├── evaluation_attention.py    # Attention model evaluation
├── main.py                    # Entry point for vanilla models
├── main_attention.py          # Entry point for attention models
├── seq2seq.py                 # Vanilla seq2seq model
├── seq2seq_attention.py       # Seq2seq with attention
├── sweep.py                   # W&B sweep for vanilla models
├── sweep_attention.py         # W&B sweep for attention models
├── train.py                   # Training for vanilla models
├── train_attention.py         # Training for attention models
├── utils.py                   # Utility functions
└── VisualizationUtils.py      # Visualization code
```

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch numpy pandas matplotlib seaborn tqdm wandb
   ```

4. **Download the Dakshina dataset:**
   - Download from [here](https://github.com/google-research-datasets/dakshina)
   - Extract to `dakshina_dataset_v1.0` in the project root

5. **Set up Weights & Biases:**
   ```bash
   wandb login
   wandb init
   ```

6. **Check and update `config.py`:**
   - Set `base_dir` and dataset paths if needed
   - Update W&B parameters:
     ```python
     wandb_project = "dakshina_transliteration"
     wandb_entity = "<your-wandb-username>"
     ```

---

## Usage

### Training (Vanilla Model)

```bash
python main.py --mode train
# For a different language:
python main.py --mode train --language ta
# With custom parameters:
python main.py --mode train --embed_size 128 --hidden_size 256 --cell_type LSTM --dropout 0.3 --batch_size 64 --learning_rate 0.001 --epochs 30
```

### Training (Attention Model)

```bash
python main_attention.py --mode train
# For a different language:
python main_attention.py --mode train --language ta
# With custom parameters:
python main_attention.py --mode train --embed_size 128 --hidden_size 256 --cell_type LSTM --dropout 0.3 --batch_size 64 --learning_rate 0.001 --epochs 30
```

### Hyperparameter Sweeps

- **Vanilla:**  
  ```bash
  python main.py --mode sweep --language hi --count 50
  ```
- **Attention:**  
  ```bash
  python main_attention.py --mode sweep --count 50
  ```

### Evaluation

- **Vanilla:**  
  ```bash
  python main.py --mode evaluate --sweep_id <sweep-id>
  ```
- **Attention:**  
  ```bash
  python main_attention.py --mode evaluate --sweep_id <sweep-id>
  ```

---

## Configuration

- All settings are managed in [`config.py`](config.py).
- Override any parameter via command-line arguments in `main.py` or `main_attention.py`.

---

## Visualization & Analysis

- Generates accuracy metrics, error analysis, and visualizations.
- Results saved in `predictions/` and logged to W&B if enabled.

---

## W&B Integration

- Tracks experiments, sweeps, and results.
- Logs metrics, models, and visualizations.

---

## References

- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)
- WanDB Report Link : [W&B Project Report](https://wandb.ai/cs24m023-indian-institute-of-technology-madras/Deep_Learning_Assignment_3/reports/DA6401-Assignment-3--VmlldzoxMjg0MzA1MQ)
- GitHub Repo Link : [GitHub Repository](https://github.com/lokeshtalamala1/Deep_Learning_Assignment_3)

---
