import os
import joblib
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import EnsembleVoteClassifier 
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV , StratifiedKFold , RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from help import load_config ,  save_model_comparison
from credit_fraud_utils_data import *
from credit_fraud_utils_eval import *


def Logisticregression(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path):
    best_param={}
    if trainer['trainer']['Logistic_Regression']['grid_search']==True:
        param_grid={
                    'C':[0.1, 1.0, 10.0],
                    'penalty':      ['l2'],
                    'class_weight': ['balanced', None, {0: 0.35, 1: 0.65}, {0: 0.25, 1: 0.75}, {0: 0.15, 1: 0.85}],
                    'solver':       ['sag', 'lbfgs', 'saga', ' newton-cg'],  
                    'max_iter':     [400, 500, 600, 800],   
        }
        log=LogisticRegression()
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        score=make_scorer(f1_score,pos_label=1)
        grid=GridSearchCV(log,param_grid,cv=skf,scoring=score,n_jobs=-1)
        grid.fit(x_train,y_train)
        best_param=grid.best_params_
        print('best parameter :',best_param)
    
    else:
        best_param=trainer['trainer']['Logistic_Regression']['parameters']

    log=LogisticRegression(**best_param,random_state=randomseed)
    log.fit(x_train,y_train)

    model_comparison , optimal_threshold = evaluate_model(log, model_comparison, path, 'Logisticregression', x_train,y_train,x_val,y_val, trainer['evaluation'])

    return {"model": log ,  "parameters": best_param, "threshold": optimal_threshold}
    


def RandomForestclassifier(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path):
    best_param={}
    if trainer['trainer']['Random_forest']['Randomized_Search']==True:
        param_grid={
            'n_estimators': [200, 400, 600 ,800],
            'min_samples_leaf': [2, 5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'class_weight': [{0: 0.20, 1: 0.80}, 'balanced_subsample', {0: 0.15, 1: 0.85}],
        }
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=randomseed)
        scorer = make_scorer(f1_score, pos_label=1)

        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(n_jobs=-1, bootstrap=True, random_state=randomseed),
            param_distributions=param_grid,
            scoring=scorer,
            cv=stratified_kfold,
            n_iter=20,  
            n_jobs=-1,
            verbose=2,
            random_state=randomseed 
        )

        random_search.fit(x_train, y_train)

        best_param = random_search.best_params_
        print("Best Hyperparameters for Random Forest:", best_param)
        
    else:
        best_param=trainer['trainer']['Random_forest']['parameters']
    rd=RandomForestClassifier(**best_param,random_state=randomseed)
    rd.fit(x_train,y_train)

    model_comparison , optimal_threshold = evaluate_model(rd, model_comparison, path, 'Random_forest', x_train,y_train,x_val,y_val, trainer['evaluation'])

    return {"model": rd ,  "parameters": best_param, "threshold": optimal_threshold}

def Neural_Network(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path):
    best_param={}
    if trainer['trainer']['Neural_Network']['Randomized_Search']==True:
        param_grid={
                'activation': ['relu'],
            'hidden_layer_sizes': [
                (30, 20), 
                (30, 20, 10), 
                (40, 30, 20), 
                (64, 32, 16),
                (64, 32, 32, 16)
            ],
            'solver': ['adam', 'sgd'],
            'batch_size': [64, 128, 512],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.001, 0.01, 0.025],
            'max_iter': [500, 800, 1000, 2000],
            'random_state': [randomseed]
        }
        Neural=MLPClassifier()
        skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=randomseed)
        score=make_scorer(f1_score,pos_label=1)
        grid=RandomizedSearchCV(Neural,param_distributions=param_grid,cv=skf,scoring=score,n_jobs=-1,n_iter=30,random_state=randomseed)
        grid.fit(x_train,y_train)
        best_param=grid.best_params_
        print('best parameter :',best_param)
    else:
        best_param=trainer['trainer']['Neural_Network']['parameters']
        Neural=MLPClassifier(
            hidden_layer_sizes=eval(best_param['hidden_layer_sizes']),
            activation=best_param['activation'],
            solver=best_param['solver'],
            alpha=best_param['alpha'],
            learning_rate_init=best_param['learning_rate_init'],
            batch_size=best_param['batch_size'],
            max_iter=best_param['max_iter'],
            random_state=randomseed)
    Neural.fit(x_train,y_train)

    model_comparison , optimal_threshold = evaluate_model(Neural, model_comparison, path, 'Neural_Network', x_train,y_train,x_val,y_val, trainer['evaluation'])

    return {"model": Neural ,  "parameters": best_param, "threshold": optimal_threshold}
    
def train_knn(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path):

    if trainer['trainer']['KNN']['grid_search'] == True:
        param_distributions = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }

        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=randomseed)
        scorer = make_scorer(f1_score, pos_label=1)

        random_search = RandomizedSearchCV(
            estimator=KNeighborsClassifier(n_jobs=-1),
            param_distributions=param_distributions,
            scoring=scorer,
            cv=stratified_kfold,
            n_iter=20,  
            n_jobs=-1,
            verbose=2,
            random_state=randomseed 
        )

        random_search.fit(x_train, y_train)

        parameters = random_search.best_params_
        print("Best Hyperparameters for KNN:", parameters)
    else:
        parameters = trainer['trainer']['KNN']['parameters']


    knn = KNeighborsClassifier(**parameters, n_jobs=-1)

    knn.fit(x_train, y_train)

    model_comparison , _ =  evaluate_model(knn, model_comparison, path, 'KNN', x_train,y_train,x_val,y_val, trainer['evaluation'])

    return {"model": knn , "parameters": parameters}
def voting_classifier(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,models,path):
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)  # make scaler learn statistics from training data
    param = trainer['trainer']['Voting_Classifier']['parameters']

    # Ensure models are present in the provided dictionary
    required_models = ['Logistic_Regression', 'Neural_Network', 'Random_forest']
    missing_models = [model for model in required_models if model not in models]
    
    if missing_models:
        raise ValueError(f"The following required models are missing: {', '.join(missing_models)}")

    try:
        voting_classifier = EnsembleVoteClassifier(
            clfs=[
                models['Logistic_Regression']['model'],
                models['Neural_Network']['model'],
                models['Random_forest']['model'],
            ],
            weights=param['weights'],
            fit_base_estimators=param['fit_base_estimators'],
            use_clones=param['use_clones'],
            voting=param['voting'],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the voting classifier: {e}")


    voting_classifier.fit(x_train, y_train) #  no refiting required here

    model_comparison, optimal_threshold = evaluate_model(voting_classifier, model_comparison, path, 'voting_classifier', x_train,y_train,x_val,y_val, trainer['evaluation'])

    return {"model": voting_classifier , "parameters": param, "threshold": optimal_threshold}

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Training')
    parser.add_argument('--data',  help='Path to data',default=('data.yml'))
    parser.add_argument('--trainer',  help='Path to trainer',default=('trainer.yml'))
    
    # Add more arguments as needed
    
    args = parser.parse_args()
    
    config_folder_path='config/'
    
    config_data=load_config( config_folder_path + args.data )
    trainer=load_config(config_folder_path+args.trainer)
    
    randomseed=config_data['randomseed']
    np.random.seed(randomseed)

    x_train,y_train,x_val,y_val=load_data(config_data)
    
    date=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    path='models/{}_under_vs_over/'.format(date)

    x_train,x_val=scale_data(x_train,config_data,val=x_val)
    
    if config_data['balanceing']['do_balance']==True:
        x_train,y_train=do_balance(x_train,y_train,config_data)

    model_comparison = {} # model comparison stats dictionary
    models = {} # trained models dictionary

    
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if trainer['trainer']['Logistic_Regression']['train']:
        models['Logistic_Regression']=Logisticregression(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path)
    
    if trainer['trainer']['Random_forest']['train']:
        models['Random_forest']=RandomForestclassifier(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path)

    if trainer['trainer']['Neural_Network']['train']:
        models['Neural_Network']=Neural_Network(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path)

    if trainer['trainer']['KNN']['train']:
        models['KNN'] = train_knn(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,path)

    if trainer['trainer']['Voting_Classifier']['train']:
        models['Voting_Classifier']=voting_classifier(x_train,y_train,x_val,y_val,randomseed,trainer,model_comparison,models,path)
    
    # Save all trained models
    model_path = path + "trained_models.pkl"
    joblib.dump(models, model_path)
    print('All models saved at: {}'.format(model_path))
    print('Evaluation plots saved at: {}evaluation/plot'.format(path))

    if model_comparison:
        # Save the model comparison plot
        model_comparison_path = path + "model_comparison-(validation dataset).png"
        save_model_comparison(model_comparison, model_comparison_path)

        print('\nModels comparison:\n')
        print(pd.DataFrame(model_comparison).T.to_markdown())
    else:
        print("No models were trained. Check your trainer configuration.")


    

    




    


    






