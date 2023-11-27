### Data Analysis on Students' grades ###
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn import feature_selection

os.chdir("C:\\Users\\mathe\\Documents\\Students_Grade_Analysis")

dfmath = pd.read_csv("student-mat.csv", delimiter=";")
dfpor = pd.read_csv("student-por.csv", delimiter=";")

df = dfmath.merge(dfpor, how = 'inner', on = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])

#idx = np.where(df.dtypes == 'int64')[0]
#idxc = np.where(df.dtypes == 'object')[0]

#pca = PCA(n_components = 30)
#pca.fit(df.iloc[:,idx])
#dfpca = pca.transform(df.iloc[:,idx])
#dfpca = pd.DataFrame(dfpca)

#pd.concat([dfpca,df.iloc[:,idxc]], axis = 1)


#df = pd.concat([dfmath,dfpor])
#
#colsx = df.columns[df.columns != "G3"]
#X = df[colsx]
#y = df["G3"]


df.drop(['G1_x','G1_y','G2_x','G2_y'], axis = 1, inplace = True)
X = df.drop(['G3_x','G3_y'], axis = 1)

for col in X.columns :
    if X[col].dtypes == "object" :
        X[col] = X[col].factorize()[0]


pca = PCA(n_components= 30)
dfpca = pca.fit_transform(X)
dfpca = pd.DataFrame(dfpca)
dfpca = sm.add_constant(dfpca)


for col in X.columns :
    if X[col].dtypes == "object" :
        X[col] = X[col].astype('category')

X = sm.add_constant(X)



#model_ols_red = sm.OLS(y, X.drop(['G1','G2','Fjob', 'Mjob', 'sex','Pstatus', 'famsup', 'guardian', 'Walc', 'activities', 'absences', 'nursery', 'traveltime', 'Fedu', 'health', 'freetime', 'address', 'reason', 'Dalc', 'famsize'], axis = 1)).fit()
model_ols_math, model_ols_por = sm.OLS(df.G3_x, X).fit(), sm.OLS(df.G3_y, X).fit()
model_ols = sm.OLS(df.G3_x, dfpca).fit()
print(model_ols_math.summary())
print(model_ols_por.summary())
print(model_ols.summary())


clf = LinearRegression()
sfs1 = sfs(clf, forward = False, floating = True, scoring = 'r2')
sfs1.fit(dfpca, df.G3_x)

dfpca.columns = np.arange(0,31)
dfpca.columns = dfpca.columns.astype(str)


max_avg_cvscore = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T.avg_score.reset_index().max()
best_cols = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T.feature_idx[16]
best_cols = pd.Series(best_cols).astype(str)

model_ols_bckw_elim = sm.OLS(df.G3_x, dfpca[best_cols]).fit()
print(model_ols_bckw_elim.summary())

f_regression(dfpca, df.G3_x)


model = feature_selection.SelectKBest(score_func=feature_selection.f_regression)
results = model.fit(dfpca, df.G3_x)
idx = np.where(results.get_support() == True)[0]
col_features = dfpca.columns[idx]

#pd.DataFrame.from_dict(sfs1.get_metric_dict()).T

model_ols_red = sm.OLS(df.G3_x, dfpca[col_features]).fit()
print(model_ols_red.summary())

#cl = ['age','failures']


model_ols = sm.OLS(y, X).fit()

anova_res = anova_lm(model_ols_red, model_ols)
print(anova_res)