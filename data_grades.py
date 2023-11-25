### Data Analysis on Students' grades ###
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

os.chdir("C:\\Users\\mathe\\Documents\\Students_Grade_Analysis")

dfmath = pd.read_csv("student-mat.csv", delimiter=";")
dfpor = pd.read_csv("student-por.csv", delimiter=";")

df = pd.concat([dfmath,dfpor])

colsx = df.columns[df.columns != "G3"]
X = df[colsx]
y = df["G3"]

for col in X.columns :
    if X[col].dtypes == "object" :
        X[col] = X[col].factorize()[0]



model_ols_red = sm.OLS(y, X.drop(['G1','G2','Fjob', 'Mjob', 'sex','Pstatus', 'famsup', 'guardian', 'Walc', 'activities', 'absences', 'nursery', 'traveltime', 'Fedu', 'health', 'freetime', 'address', 'reason', 'Dalc', 'famsize'], axis = 1)).fit()
print(model_ols_red.summary())


clf = LinearRegression()
sfs1 = sfs(clf,forward=False, scoring='r2')
sfs1.fit(X.drop(["G2","G1"], axis = 1),y)

sfs1.k_feature_names_


#cl = ['age','failures']


model_ols = sm.OLS(y, X).fit()

anova_res = anova_lm(model_ols_red, model_ols)
print(anova_res)