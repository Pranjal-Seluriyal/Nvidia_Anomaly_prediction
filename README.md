# GPU-HealthNet 🚀
### LSTM-Based GPU Anomaly Detection

## 🔍 Overview
GPU-HealthNet is a machine learning project that monitors GPU telemetry
(temperature, power, utilization, memory) and detects abnormal behavior
using an LSTM autoencoder.

Instead of predicting rare hardware failures, the system learns
**normal GPU behavior** and flags deviations as anomalies.

## ⚙️ How It Works
1. GPU telemetry is collected using NVIDIA `nvidia-smi`
2. Time-series data is preprocessed and normalized
3. Sliding windows are created for sequence learning
4. An LSTM autoencoder learns normal behavior
5. High reconstruction error indicates abnormal GPU behavior

## ▶️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook gpu_anomaly_detection.ipynb
```

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- NVIDIA nvidia-smi

## 📌 Disclaimer
This project detects abnormal GPU behavior, not physical hardware failure.
