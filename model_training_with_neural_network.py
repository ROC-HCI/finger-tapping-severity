import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import wandb
import random
import click
import imblearn

from tqdm import tqdm
from pandas import DataFrame
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from shaphypetune import BoostSearch, BoostRFE, BoostRFA, BoostBoruta
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, r2_score, mean_absolute_percentage_error, accuracy_score
from imblearn.over_sampling import SMOTE

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

RATING_FILE = "./severity_dataset_dropped_correlated_columns.csv"

def load():
    df = pd.read_csv(RATING_FILE)
    features = df.loc[:, 'wrist_mvmnt_x_median':'acceleration_min_trimmed']
    labels = df["Rating"]

    def parse(name:str):
        if name.startswith("NIH"): [ID, *_] = name.split("-")
        else: [*_, ID, _, _] = name.split("-")
        return ID
    
    df["id"] = df.filename.apply(parse)

    return features, labels, df["id"]

def select(features:DataFrame, labels, **cfg):
    methods = { "BoostRFE":BoostRFE, "BoostRFA":BoostRFA, "BoostBoruta":BoostBoruta }

    SELECTOR = methods[cfg["selector"]]

    base = XGBRegressor() if cfg["selector_base"] == "XGB" else LGBMRegressor()
    
    selector = SELECTOR(base)
    selector.fit(features, labels)

    sorts = selector.ranking_.argsort()
    selected = features.columns[sorts][:cfg["n"]]
    features = features[selected]

    return features, labels


def metrics(preds, labels):
    results = {}

    preds, labels = onp.array(preds), onp.array(labels)
    results["mae"] = mean_absolute_error(labels, preds)
    results["mse"] = mean_squared_error(labels, preds)
    results["r2"] = r2_score(labels, preds)
    results["mape"] = mean_absolute_percentage_error(labels + 1, preds + 1) # shift labels by 1
    results["pearsonr"], _ = stats.pearsonr(labels, preds)

    rounded_preds = onp.round(preds)
    rounded_labels = onp.round(labels)
    results["kappa.no.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights=None)
    results["kappa.linear.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="linear")
    results["kappa.quadratic.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="quadratic")
    results["accuracy"] = accuracy_score(rounded_labels, rounded_preds)
    results["kenndalltau"], _ = stats.kendalltau(labels, preds)
    results["spearmanr"], _ = stats.spearmanr(labels, preds)

    return results

class ANN(nn.Module):
    def __init__(self, n_features):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=(int)(n_features/2), bias=True)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=1,bias=True)
        self.hidden_activation = nn.ReLU()
        self.final_activation = nn.ReLU()
    
    def forward(self,x):
        x1 = self.hidden_activation(self.fc1(x))
        y = self.final_activation(self.fc2(x1))
        return y

class ShallowANN(nn.Module):
    def __init__(self, n_features):
        super(ShallowANN, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1,bias=True)
        self.activation = nn.ReLU()
    def forward(self,x):
        y = self.activation(self.fc(x))
        return y
    
class MyDataset(Dataset):
    def __init__(self,features,labels):
        self.features = torch.Tensor(onp.asarray(features))
        self.labels = torch.Tensor(labels).reshape(-1)
        self.n_samples = len(labels)
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return self.n_samples

@click.command()
@click.option("--model", default="ANN", help="Options: ShallowANN, ANN")
@click.option("--selector", default="BoostRFE", help="Feature selection method")
@click.option("--selector_base", default="LGBM", help="Base regressor for feature selection")
@click.option("--learning_rate", default=0.01313, help="Learning rate for regressor")
@click.option("--n", default=22, help="Number of features to select")
@click.option("--random_state", default=42, help="Random state for regressor")
@click.option("--seed", default=42, help="Seed for random")
@click.option("--use_feature_selection",default='no',help="yes if you want to select features, no if you want to work with all features")
@click.option("--use_feature_scaling",default='no',help="yes if you want to scale the features, no otherwise")
@click.option("--scaling_method",default='StandardScaler',help="Options: StandardScaler, MinMaxScaler")
@click.option("--minority_oversample",default='no',help="Options: yes, no")
@click.option("--batch_size",default=32,help='number of samples per iteration')
@click.option("--num_epochs",default=10,help="number of epochs")
def main(**cfg):
    wandb.init(project="npj-severity-paper", config=cfg)
    features, labels, ids = load()

    if(cfg["use_feature_selection"]=='yes'):
        features, labels = select(features, labels, **cfg)
    
    all_preds = []
    all_labels = []
    save_preds = pd.read_csv(RATING_FILE)
    save_preds["preds"] = save_preds["Rating"]
    loo = LeaveOneGroupOut()
    oversample = SMOTE(random_state = cfg['random_state'])
    criterion = torch.nn.MSELoss()

    for train_index, test_index in tqdm(loo.split(features, groups=ids), total=len(ids.unique())):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        if cfg['use_feature_scaling']=='yes':
            if cfg['scaling_method'] == 'StandardScaler':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if cfg['minority_oversample']=='yes':
            y_train[y_train==4] = 3
            y_test[y_test==4] = 3
            (X_train, y_train) = oversample.fit_resample(X_train, y_train)

        y_train = onp.asarray(y_train)
        y_test = onp.asarray(y_test)
        
        train_dataset = MyDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
        test_dataset = MyDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'])
        
        model = ANN(features.shape[1])
        optimizer = torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'])

        for epoch in range(cfg['num_epochs']):
            for i, (x, y) in enumerate(train_loader):
                y_preds = model(x).reshape(-1)
                l = criterion(y_preds,y)
                l.backward()
                optimizer.step()
                optimizer.zero_grad()
            
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                y_preds = model(x).reshape(-1)
                
                #print(y_preds.shape, y.shape)
                all_preds.extend(y_preds.numpy())
                all_labels.extend(y.numpy())
                
                save_preds["preds"].iloc[test_index] = y_preds.numpy().reshape(-1)

    results = metrics(all_preds, all_labels)
    wandb.log(results)
    #wandb.log({"predictions":wandb.Table(dataframe=save_preds)})

if __name__ == "__main__":
    main()
