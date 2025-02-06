# Fraud

Start-to-production implementation of bank fraud detection using XGBoost

## Why XGBoost?

Most of the time, decision tree based approaches work better with structured,
tabular data. More importantly, compared to deep learning approaches, they are
better fit for imbalanced data like fraud data.

`XGBoost` is the go-to library when it comes to implementing gradient boosting
algorithms

## Build and Run

With `uv` installed

```bash
## 1. Clone the repo

git clone git@github.com:hmzdot/fraud-ml.git
cd fraud-ml

## 2. Install dependencies

# Either with uv
uv sync

# Or with pip (assuming you've setup venv yourself)
pip3 install .

## 3. Train the model

# Either with uv
uv run src/train.py

# Or with python3 (assuming venv is active)
python3 src/train.py

## 4. Evaluate another dataset

# Either with uv
uv run src/eval.py ./snapshots/{model} {path/to/data}

# Or with python3
python3 src/eval.py ./snapshots/{model} {path/to/data}
```
