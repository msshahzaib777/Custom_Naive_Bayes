# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:50:33 2020

@author: muham
"""
import pandas as pd
import numpy as np
import math
import copy

def continous(X, mean, variance):
    a = (1/np.sqrt(2*math.pi*variance))
    b = np.exp(-(np.power(X-mean, 2)/(2*variance)))
    return a*b

dataset = pd.read_csv("data.txt", header=None, names=["Refund", "Martial", "Income", "Evade"])
train = dataset.iloc[:10, :]
X_test = dataset.iloc[10:, :-1].reset_index(drop=True)
X = train[["Refund", "Martial", "Income"]]
y = train["Evade"]


def fit(X, y):
    NaiveBayes = dict()
    NaiveBayes["PrClass"] = y.value_counts()
    data = copy.deepcopy(X)
    data.insert(3, "y", y)
    for i in data.columns.values[:-1]:
        if(data.loc[:, i].dtype == 'O'):
            index = [data[i].unique(), y.unique()]
            multiindex = pd.MultiIndex.from_product(index, names=[i, y.name])
            model = pd.DataFrame(index=multiindex)
            model.insert(0, "Value", 0)
            for j in data[i].unique():
                for k in y.unique():
                    model.loc[j, k] = (data.where((data[i] == j) & (data['y'] == k)).count()[0])
            NaiveBayes[i] = model
        else:
            model = pd.DataFrame(index=y.unique())
            model.insert(0, "Mean", 0)
            model.insert(1, "Variance", 0) 
            for k in y.unique():
                mean = np.mean(data[data['y'] == k])
                model.loc[k, "Mean"] = mean.iloc[0]
                var = np.var(data[data['y'] == k])
                model.loc[k, "Variance"] = var.iloc[0]
            NaiveBayes[i] = model
    return NaiveBayes

def predict(X, model):
    lst = []
    for k, r in X.iterrows():
        scores = pd.DataFrame(index= model["PrClass"].index.values)
        scores.insert(0, "Score", 0)
        for i in model["PrClass"].index.values:
            neu = 1
            for j in r.index.values:
                if(type(r[j]) == str):
                    neu = neu * (model[j].loc[r[j], i]/ model["PrClass"].loc[i])
                else:
                    neu = neu * (continous(r[j], model[j].loc[i, 'Mean'], model[j].loc[i, 'Variance']))
            neu = neu * (model["PrClass"].loc[i]/model["PrClass"].sum())
            scores.loc[i, "Score"] = neu[0]
        lst.append(scores.idxmax()[0])
    return lst
    

model = fit(X, y)
predict(X_test, model)


dataset2 = pd.read_csv("data2.txt", header=None, names=["Refund", "Status", "Tax", "Cheat"], sep=' ') 
train = dataset2.iloc[:13, :]
X_test = dataset2.iloc[13:, :-1].reset_index(drop=True)
X = train[["Refund", "Status", "Tax"]]
y = train["Cheat"]

model = fit(X, y)
predict(X_test, model)

temp = train[train["Cheat"] == "No"]
x=63
print(continous(x, 99.285, 1261.9))
print(continous(x, 75, 60))
