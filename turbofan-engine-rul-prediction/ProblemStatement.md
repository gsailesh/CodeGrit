# Mission: Turbofan Engine Remaining Useful Life (RUL) Prediction

## Skill Target
**Time Series Forecasting & Predictive Maintenance** — a critical industrial ML skill covering sensor data processing, sequence modeling with PyTorch, and real-world degradation analysis.

## Data URL
- **Primary:** [NASA C-MAPSS Turbofan Engine Degradation Dataset (Kaggle)](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- **Reference Paper:** [Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation (NASA)](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## About The Dataset
```md
- `train_FD00X.txt` (Run-to-Failure): This is your training data. Every single engine in these files runs until the moment it dies. You know exactly when they failed, which makes it perfect for training your model.

- `test_FD00X.txt` (Run-to-Prior-to-Failure): This is your test data. The data for these engines is cut off early before they fail. Your model's job is to look at this cut-off data and guess: "How many more cycles does this engine have left?"

- `RUL_FD00X.txt` (Remaining Useful Life): This is the answer key for the test set! It stands for Remaining Useful Life. If Engine #1 in test_FD001.txt gets cut off at cycle 31, the first row in RUL_FD001.txt might say 112. That means Engine #1 actually had 112 cycles left before it failed.

- `x.txt`: This is just a leftover/accidental file included by whoever uploaded the dataset to Kaggle. Ignore this file.
```

## Objectives
1. **Data Ingestion & EDA:** Load the C-MAPSS FD001 subset (single operating condition, single fault mode). Explore sensor correlations, engine degradation curves, and operational regime distributions using Polars and Matplotlib/Plotly.
2. **Feature Engineering:** Construct a piecewise-linear RUL target (capped at 125 cycles). Apply rolling-window statistics and sensor normalization across the fleet.
3. **Baseline Model:** Train a classical regression baseline (e.g., Random Forest or XGBoost) on hand-crafted features to establish a benchmark RMSE.
4. **Sequence Model (PyTorch):** Build an LSTM or 1D-CNN model in PyTorch that takes sliding windows of raw sensor readings and predicts RUL. Compare against the baseline.
5. **Evaluation & Visualization:** Plot predicted vs. actual RUL curves per engine. Compute the NASA scoring function alongside RMSE. Build a simple Streamlit dashboard showing fleet health status.

## Critical Design Question (Puzzle)
*The NASA scoring function penalizes late predictions (predicting an engine is healthy when it's about to fail) far more severely than early predictions. How should this asymmetric cost function influence your model's loss function during training, and what are the trade-offs of embedding domain-specific costs directly into gradient-based optimization vs. using a symmetric loss and adjusting thresholds post-hoc?*

## Community Insights

### Pre-processing & Feature Engineering
- **Constant Sensor Removal:** Identified and removed 7 sensors with zero variance (s1, s5, s6, s10, s16, s18, s19).
- **Piecewise Linear RUL:** Capped target RUL at 125 cycles to better model the "stable-then-degrade" nature of engine health.
- **Normalization:** Used `MinMaxScaler` fitted exclusively on training data to prevent data leakage.

### Architecture Trends
- **Bi-LSTM + Attention:** Top models use Bi-Directional LSTMs with an attention mechanism to weigh critical failure-prone cycles more heavily.
- **Hybrid 1D-CNN/LSTM:** Using 1D-CNN layers for automated feature extraction followed by LSTM for temporal memory.
- **Deep-Tree Hybrids:** Extracting embeddings from a neural network and using them as input for an XGBoost regressor for final refinement.

### AHA! Factor
- **SHAP for Sensors:** Using SHAP (Shapley Additive Explanations) revealed that ~10 sensors carry 90% of the predictive power, allowing for leaner, more robust models.

## Experiments to Try
- **Asymmetric Loss:** "Model safety" experiment. Train with a loss function that penalizes late predictions (RUL overestimation) more severely than early ones.
- **Attention Heatmaps:** Visualize attention weights to confirm if the model is correctly focusing on the degradation phase.
- **Residual Feature Engineering:** Calculate residuals between actual sensor readings and a "healthy-state" baseline as a sensitive indicator of early-stage degradation.
