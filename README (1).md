
# 💳 Credit Card Fraud Detection 🚨

This project builds a robust system to **detect fraudulent credit card transactions** using classical machine learning models and a deep learning model with **Focal Loss**. It includes complete workflows for data processing, model training, evaluation, and reporting.

---

## 📁 Project Structure

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

## 📊 Models Used

- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors**
- **Neural Network (MLPClassifier)**
- **Voting Classifier (Ensemble)**
- **Custom Neural Network with Focal Loss (PyTorch)**

---

## ⚙️ Configuration Files

- `config/data.yml` – Paths, balancing, scaling
- `config/trainer.yml` – Training settings for each model, evaluation config

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ Includes `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `torch`, `mlxtend`

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

## 📈 Evaluation Outputs

Each trained model is saved in `models/<timestamp>/` with:

- Classification Reports
- Confusion Matrix Heatmaps
- Precision-Recall Curves
- Threshold-based tuning results
- Voting classifier ensemble (if enabled)
- PR-AUC metrics

---

## 🧠 Focal Loss Model (PyTorch)

Trained via custom architecture with:

```python
BatchNorm → Tanh → Dropout → Focal Loss
```

Designed to handle extreme class imbalance with tunable `alpha` and `gamma`.

---

## 📉 Class Imbalance Handling

Supports multiple techniques:
- Class Weights (for LR, RF)
- Oversampling (`do_balance`)
- Focal Loss (for PyTorch NN)

---

## 📊 Example Result (Markdown Table)

| Model              | F1 Score (Positive) | Precision | Recall | PR-AUC |
|-------------------|---------------------|-----------|--------|--------|
| LogisticRegression| 0.84                | 0.86      | 0.82   | 0.88   |
| Random Forest      | 0.87               | 0.85      | 0.89   | 0.90   |
| FocalLoss NN       | 0.90               | 0.88      | 0.92   | 0.94   |

---

## ✅ Future Enhancements

- 🔁 Early Stopping & Best Checkpoint saving
- 📦 Packaging as pip-installable module
- 📉 Support for ROC-AUC evaluation
- 📈 Hyperparameter tuning automation via Optuna
- 📤 Model deployment as Flask/FastAPI service

---

## 👨‍💻 Author

**Mostafa HagAli**  
*Machine Learning Engineer*  
📧 your_email@example.com  
🔗 [LinkedIn/GitHub/Portfolio]

---

## 🛡️ License

MIT License - Feel free to use, modify, and distribute.

---
