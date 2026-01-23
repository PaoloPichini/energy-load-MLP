# Hourly Energy Consumption Forecasting (MLP Baseline)

This project trains a simple multilayer perceptron (MLP) to predict the next
hour's load (MW) from a sliding window of past hourly values in the
`AEP_hourly.csv` series.

## Project Structure
- `energy-load-MLP.ipynb`: Main notebook for data
  loading, windowing, training, and evaluation.
- `config1.json`: Hyperparameters (window length, model size, training settings).
- `requirements.txt`: Python dependencies.

## Setup
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
The notebook uses `kagglehub` to download:
`robikscube/hourly-energy-consumption`. The dataset is loaded via:

```python
path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
df = pd.read_csv(f"{path}/AEP_hourly.csv")
```

## Method (Sliding Window)
Given a window length `L`, each input is the last `L` hourly MW values and the
target is the next MW value. This creates many (X, y) pairs from a single series.

## Training
- Train/val/test split is time-ordered (70/15/15).
- Inputs are standardized with train mean/std.
- MLP is trained with MSE loss and Adam optimizer.

## Results
The notebook reports MSE/RMSE/MAE on the test split and can plot predictions vs
targets.

## Notes
- Ensure `input_dim` in `config1.json` matches `window_length`.
- For reproducibility, consider setting a random seed in the notebook.
