
#  Credit Card Fraud Detection 

This project builds a robust system to **detect fraudulent credit card transactions** using classical machine learning models and a deep learning model with **Focal Loss**. It includes complete workflows for data processing, model training, evaluation, and reporting.

---

## Project Structure

```
CREDIT-CARD-FRAUD-DETECTION/
├── config/                     # Configuration YAMLs
│   ├── data.yml
│   └── trainer.yml
├── data/                       # Input CSV files
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── trainval.csv
├── models/                     # Saved models and evaluation results
│   ├── <timestamped folders>/
│   └── focal_loss_checkpointsfocal_last/
├── credit_fraud_train.py       # Training for classical ML models
├── Focal_loss.py               # Training PyTorch model with Focal Loss
├── test_script.py              # Evaluation on test set
├── credit_fraud_utils_data.py  # Data loading, scaling, balancing
├── credit_fraud_utils_eval.py  # Evaluation metrics and plots
├── help.py                     # Helper functions (config/model IO)
├── EDA.ipynb                   # Exploratory Data Analysis notebook
└── README.md                   # 📄 You are here
```

---

## Models Used

- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors**
- **Neural Network (MLPClassifier)**
- **Voting Classifier (Ensemble)**
- **Custom Neural Network with Focal Loss (PyTorch)**

---

## Configuration Files

- `config/data.yml` – Paths, balancing, scaling
- `config/trainer.yml` – Training settings for each model, evaluation config

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> Includes `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `torch`, `mlxtend`

---

### 2. Train Models (Classical ML)
```bash
python credit_fraud_train.py --data data.yml --trainer trainer.yml
```

---

### 3. Train Deep Model (Focal Loss)
```bash
python Focal_loss.py
```

---

### 4. Evaluate on Test Set
```bash
python test_script.py
```

---

## Evaluation Outputs

Each trained model is saved in `models/<timestamp>/` with:

- Classification Reports
- Confusion Matrix Heatmaps
- Precision-Recall Curves
- Threshold-based tuning results
- Voting classifier ensemble (if enabled)
- PR-AUC metrics

![Result]("models\2025_07_12_13_20\model_comparison-(validation dataset).png")

---

## Focal Loss Model (PyTorch)

Trained via custom architecture with:

```python
BatchNorm → Tanh → Dropout → Focal Loss
```

Designed to handle extreme class imbalance with tunable `alpha` and `gamma`.

---

## Class Imbalance Handling

Supports multiple techniques:
- Class Weights (for LR, RF)
- Oversampling (`do_balance`)
- Focal Loss (for PyTorch NN)

---


