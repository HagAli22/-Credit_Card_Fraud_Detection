import torch
import torch.nn as nn
import torch.nn.functional as F
from credit_fraud_utils_data import *
from help import *
from credit_fraud_utils_eval import *
from torch.utils.tensorboard import SummaryWriter  

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        # focal loss : https://leimao.github.io/blog/Focal-Loss-Explained/
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_logits, target):
        BCELoss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        prob = pred_logits.sigmoid()
        alpha_t = torch.where(target == 1, self.alpha, (1 - self.alpha))
        pt =  torch.where(target == 1, prob, 1 - prob)
        loss = alpha_t * ((1 - pt) ** self.gamma) * BCELoss
        return loss.sum()

class FraudDetectionNN(nn.Module):
    def __init__(self):
        super(FraudDetectionNN, self).__init__()
        self.hidden1 = nn.Linear(30, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.hidden2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.hidden3 = nn.Linear(128, 16, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 1)
        self.tanh = nn.Tanh()
        self.dorpout = nn.Dropout(0.5)
    

    def forward(self, x):
        x = self.tanh(self.bn1(self.hidden1(x)))
        x = self.dorpout(x)
        x = self.tanh(self.bn2(self.hidden2(x)))
        x = self.dorpout(x)
        x = self.tanh(self.bn3(self.hidden3(x)))
        x = self.output(x)
        return x

if __name__ == '__main__':

    config = load_config("config/data.yml")
    torch.manual_seed(config['randomseed']) 

    X_train, y_train, X_val, y_val = load_data(config)
    X_train, X_val = scale_data(X_train,config,val=X_val)

    # uncomment if you want to balance data using-over-sampling
    # print("Nig: ", len(y_train[y_train == 0])," Pos: ",len(y_train[y_train == 1]))
    # X_train, y_train =do_balance(X_train,y_train,config)
    # print("Nig: ",len(y_train[y_train == 0])," Pos: ",len(y_train[y_train == 1]))

 
    model = FraudDetectionNN()
    alpha = 0.75 # (rate to make balance class)                
    gamma = 2 # (focusing on hard samples "minority class") 
    lr = 0.001

    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer =  torch.optim.SGD(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train ,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    batch_size = 512 # 1024 2048 4096
    num_epochs = 450
    start_epoch = 0

    # # Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        
        # shuffle training data
        permutation = torch.randperm(X_train_tensor.size()[0])
        X_train_tensor_shuffled = X_train_tensor[permutation].clone()
        y_train_tensor_shuffled = y_train_tensor[permutation].clone()

        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor_shuffled[i:i+batch_size]
            y_batch = y_train_tensor_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # log epoch statistics
        epoch_loss /= len(X_train_tensor) / batch_size
        if (epoch + 1)>=200 and (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, epoch_loss))

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor).item()

        if (epoch + 1)>=400 and (epoch + 1)% 10 == 0:
            # ✅ احفظ آخر checkpoint فقط في ملف ثابت:
            save_checkpoint(model, epoch + 1, title='focal_last')  # سيحفظ باسم: models/focal_last_checkpoint.pth
                    

    model.eval()
    with torch.no_grad():
        val_output = model(X_train_tensor)
        y_train_prob = val_output.sigmoid().numpy()
        y_train_pred = (y_train_prob > 0.5).astype(int)
        _ = eval_classification_report_confusion_matrix(y_true=y_train, y_pred=y_train_pred, title='FraudDetectionNN train') 
        # eval_precision_recall_for_different_threshold(y_pred=y_train_prob, y_true=y_train)

        val_output = model(X_val_tensor)
        y_val_prob = val_output.sigmoid().numpy()
        y_val_pred = (y_val_prob > 0.50).astype(int)
        report_val = eval_classification_report_confusion_matrix(y_true=y_val, y_pred=y_val_pred, title='FraudDetectionNN valdtion')

        
        optimal_threshold, f1_scores = eval_best_threshold(y_pred=y_train_prob, y_true=y_train, with_repect_to="f1_score")  
        y_val_pred = (y_val_prob > optimal_threshold).astype(int)
        report_val = eval_classification_report_confusion_matrix(y_pred=y_val_pred, y_true=y_val, title='FraudDetectionNN optimal threshold')


######################################################################################################################################
# Some learning lessons & Notes:
# 1. Alpth and gamma sometimes unstables train using batchnorm make this effect less occur and switching from Adam to SGD also.     
# 2. High gamma (5~7) gives very noisey loss Curve 
# 3. Alpha is very crucial to balance the two classes (need hyperparameter tuning).
#####################################################################################################################################