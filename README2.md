# Deep_Learning_Assignment_3
Use recurrent neural networks to build a transliteration system.
-------------------------------------------------------------------
## References

- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)
- WanDB Report Link : [W&B Project Report](https://wandb.ai/cs24m023-indian-institute-of-technology-madras/Deep_Learning_Assignment_3/reports/DA6401-Assignment-3--VmlldzoxMjg0MzA1MQ)
- GitHub Repo Link : [GitHub Repository](https://github.com/lokeshtalamala1/Deep_Learning_Assignment_3)

---
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
- Source: Google Dakshina dataset (Hindi transliteration subset)
* Files:
hi.translit.predictions_attention_2.csv (with attention)
hi.translit.predictions_vanilla_2.csv (without attention)
<Devanagari>


---

## Project Overview
Task: Hindi (Latin → Devanagari) transliteration
Architecture: Encoder–Decoder with RNN variants (SimpleRNN, LSTM, GRU)
Attention: Bahdanau additive mechanism for improved alignment

---

## Features
Character-level tokenization for both scripts
Configurable RNN backbones: SimpleRNN, LSTM, GRU
Attention-based decoder for better context handling
Experiment tracking with Weights & Biases (wandb)
Evaluation metrics: exact-match accuracy, average edit distance


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

6. **Check and update `Train.py`:**
   - Set `base_dir` and dataset paths if needed
   - Update W&B parameters:
     ```python
     wandb_project = "Deep_Learning_Assignment_3"
     wandb_entity = "<cs24m023-indian-institute-of-technology-madras>"
     ```

---
## Results & Evaluation
Loss & accuracy curves in results/logs/
Sample transliterations in results/predictions/
W&B dashboard for interactive analysis


## Visualization & Analysis

- Generates accuracy metrics, error analysis, and visualizations.
- Results saved in `predictions/` and logged to W&B if enabled.

---

## W&B Integration

- Tracks experiments, sweeps, and results.
- Logs metrics, models, and visualizations.

---
