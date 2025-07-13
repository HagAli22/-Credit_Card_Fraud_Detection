import torch
from credit_fraud_utils_eval import *
from help import *
from credit_fraud_utils_data import load_test, load_data, scale_data
from sklearn.preprocessing import RobustScaler
from Focal_loss import FraudDetectionNN
import numpy as np


def evaluate_FraudDetectionNN_focal_loss(config):
    checkpoint_id = 400
    checkpoint_path = f'models/focal_loss_checkpointsfocal_last/checkpoint_epoch_{checkpoint_id}.pth'
    model = FraudDetectionNN()
    load_checkpoint(model, checkpoint_path)

    X_train, _, _, _ = load_data(config)
    X_test, y_test = load_test(config)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred_prob = y_pred_logits.sigmoid().numpy()

    # Optimal threshold
    print("\nEvaluation with optimal threshold based on F1 score:")
    optimal_threshold, _ = eval_best_threshold(y_pred=y_pred_prob, y_true=y_test)
    y_pred_opt = (y_pred_prob > optimal_threshold).astype(int)
    eval_classification_report_confusion_matrix(y_true=y_test, y_pred=y_pred_opt, title='Focal Loss NN - test ')
    


def evaluate_VotingClassifier(config, model_path):
    model = load_model(model_path)['Voting_Classifier']['model']

    X_test, y_test = load_test(config)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    

    print("\nEvaluation with optimal threshold based on F1 score:")
    optimal_threshold, _ = eval_best_threshold(y_pred=y_pred_prob, y_true=y_test)
    y_pred_opt = (y_pred_prob > optimal_threshold).astype(int)
    eval_classification_report_confusion_matrix(y_true=y_test, y_pred=y_pred_opt, title="Voting Classifier test @ optimal")
    
    print(f"Optimal threshold used: {optimal_threshold:.4f}")

    print(f"\nTrue frauds in test set: {np.sum(y_test)}")
    print(f"Detected frauds (predicted 1s): {np.sum(y_pred_opt)}")


if __name__ == "__main__":
    config_path = 'config/data.yml'
    model_path = 'models/2025_07_12_13_20/trained_models.pkl'
    config = load_config(config_path)

    # اختر واحدة للتقييم:
    evaluate_VotingClassifier(config=config, model_path=model_path)
    evaluate_FraudDetectionNN_focal_loss(config=config)
