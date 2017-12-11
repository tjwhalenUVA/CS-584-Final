# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:48:34 2017

@author: timothy.whalen
"""
#%%Set Up
print('Import Packages')
import pandas as pd
import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc

#from sklearn.manifold import TSNE

os.chdir(r'C:\Users\e481340\Documents\GMU MASTERS\CS 584\CS584_Final\source')
import helper_python as hp

os.chdir(r'C:\Users\e481340\Documents\GMU MASTERS\CS 584\CS584_Final')
mainFolder = os.getcwd()

#%% Read Data
#Raw Data
print('Read Data')
df = pd.read_csv(mainFolder + r'\data\voice.csv')
df = df.drop(['modindx', 'skew', 'maxdom'], axis=1)

#Split Data
print('Split Data')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), 
                                                    df.label, 
                                                    test_size=0.33,
                                                    random_state=0)

#tsne = TSNE(n_components=3, 
#            random_state=0).fit_transform(df.drop('label', axis=1))
#tsne = pd.DataFrame(tsne)

#%%KNN
print('KNN')
knnparams = {'n_neighbors': np.arange(3, 21, 2), 
             'weights': ['uniform', 'distance'], 
             'algorithm': ['auto', 'ball_tree', 
                           'kd_tree', 'brute'],
             'p': [1, 2]}

knnsearch = GridSearchCV(estimator=KNeighborsClassifier(), 
                          param_grid=knnparams, 
                          cv=5)
print('    Fit')
knnsearch.fit(X_train, y_train)

knn_pred = knnsearch.predict(X_test)

#Model Evaluation
knnsearch.best_score_
knnsearch.best_params_
knnAcc = accuracy_score(y_test, knn_pred)
knnCM = pd.crosstab(y_test, knn_pred, 
                    rownames=['True'], colnames=['Predicted'], 
                    margins=True)

knnGSresults = pd.DataFrame(knnsearch.cv_results_)

probs = knnsearch.predict_proba(X_test)
male_score = []
for i in probs:
    male_score.append(i[1])
fpr, tpr, threshs = roc_curve(y_test, male_score, 'male')
knn_roc = pd.DataFrame({'fpr': fpr, 
                       'tpr': tpr})
    
knn_auc = pd.DataFrame({'auc': [auc(fpr, tpr)]})

#Save Results
print('    Save Results')
knnparams = hp.bestParamDF(knnsearch)
knnresult = pd.DataFrame({'true': y_test, 
                          'pred': knn_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\knn_results.xlsx', 
                        engine='xlsxwriter')

knnparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

knnresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

knnGSresults.to_excel(writer, 
                      sheet_name='GSresult', 
                      index=False)

knn_roc.to_excel(writer, 
                      sheet_name='roc', 
                      index=False)

knn_auc.to_excel(writer, 
                      sheet_name='auc', 
                      index=False)

writer.save()

#%% Decision Tree
print('DT')
dtparams = {"max_depth": np.arange(1,30), 
            "criterion": ['gini', 'entropy'], 
            "splitter": ['best', 'random'], 
            "presort": [True, False]}

dtsearch = GridSearchCV(estimator=DecisionTreeClassifier(), 
                         param_grid=dtparams, 
                         cv=5)
print('    Fit')
dtsearch.fit(X_train, y_train)

dt_pred = dtsearch.predict(X_test)

#Model Evaluation
dtsearch.best_score_
dtsearch.best_params_
dtAcc = accuracy_score(y_test, dt_pred)
dtCM = pd.crosstab(y_test, dt_pred, 
                    rownames=['True'], colnames=['Predicted'], 
                    margins=True)

dtDF = hp.dtDFGenerator(dtsearch.best_estimator_.tree_, 
                        X_train)

hp.dtplot(dtsearch.best_estimator_, 
          fileName = 'dt', 
          X_train = X_train)

#Get ROC Curve
probs = dtsearch.predict_proba(X_test)
male_score = []
for i in probs:
    male_score.append(i[1])
fpr, tpr, threshs = roc_curve(y_test, male_score, 'male')
dt_roc = pd.DataFrame({'fpr': fpr, 
                       'tpr': tpr})

dt_auc = pd.DataFrame({'auc':[auc(fpr, tpr)]})

dtCVresult = pd.DataFrame(dtsearch.cv_results_)

#Save Results
print('    Save Results')
dtparams = hp.bestParamDF(dtsearch)
dtresult = pd.DataFrame({'true': y_test, 
                          'pred': dt_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\dt_results.xlsx', 
                        engine='xlsxwriter')

dtparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

dtresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

dtDF.to_excel(writer, 
              sheet_name='tree', 
              index=False)

dt_roc.to_excel(writer, 
                sheet_name='roc', 
                index=False)

dt_auc.to_excel(writer, 
                sheet_name='auc', 
                index=False)

dtCVresult.to_excel(writer, 
                sheet_name='CVresult', 
                index=False)

writer.save()


#%% Random Forest
print('RF')
rfparams = {"n_estimators": np.arange(5,30), 
            "max_depth": np.arange(1,15), 
            "criterion": ['gini', 'entropy']}

rfsearch = GridSearchCV(estimator=RandomForestClassifier(), 
                         param_grid=rfparams, 
                         cv=5)
print('    Fit')
rfsearch.fit(X_train, y_train)

rf_pred = rfsearch.predict(X_test)

#Model Evaluation
rfsearch.best_score_
rfsearch.best_params_
rfAcc = accuracy_score(y_test, rf_pred)
rfCM = pd.crosstab(y_test, rf_pred, 
                    rownames=['True'], colnames=['Predicted'], 
                    margins=True)

#Get ROC Curve
probs = rfsearch.predict_proba(X_test)
male_score = []
for i in probs:
    male_score.append(i[1])
fpr, tpr, threshs = roc_curve(y_test, male_score, 'male')
rf_roc = pd.DataFrame({'fpr': fpr, 
                       'tpr': tpr})
    
rf_auc = pd.DataFrame({'auc':[auc(fpr, tpr)]})

#rfDF = hp.rfDFGenerator(rfsearch.best_estimator_.tree_, 
#                        X_train)


rfCVresult = pd.DataFrame(rfsearch.cv_results_)

#Save Results
print('    Save Results')
rfparams = hp.bestParamDF(rfsearch)
rfresult = pd.DataFrame({'true': y_test, 
                          'pred': rf_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\rf_results.xlsx', 
                        engine='xlsxwriter')

rfparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

rfresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

rf_roc.to_excel(writer, 
                   sheet_name='roc', 
                   index=False)

rf_auc.to_excel(writer, 
                   sheet_name='auc', 
                   index=False)

rfCVresult.to_excel(writer, 
                     sheet_name='CVresult', 
                     index=False)

writer.save()

#%% SVM
print('SVM')
svmparams = {'C': np.arange(10, 50, 1)}

svmsearch = GridSearchCV(estimator=SVC(), 
                         param_grid=svmparams, 
                         cv=5)
print('    Fit')
svmsearch.fit(X_train, y_train)

svm_pred = svmsearch.predict(X_test)

svmAcc = accuracy_score(y_test, svm_pred)
svmCM = pd.crosstab(y_test, svm_pred, 
                    rownames=['True'], colnames=['Predicted'], 
                    margins=True)


score = svmsearch.decision_function(X_test)
score_roc = roc_curve(y_test, score, 'male')
svm_roc = pd.DataFrame({'fpr': score_roc[0], 
                       'tpr': score_roc[1]})

svm_auc = pd.DataFrame({'auc':[auc(svm_roc.fpr, svm_roc.tpr)]})

svmCVresult = pd.DataFrame(svmsearch.cv_results_)

#Save Results
print('    Save Results')
svmparams = hp.bestParamDF(svmsearch)
svmresult = pd.DataFrame({'true': y_test, 
                          'pred': svm_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\svm_results.xlsx', 
                        engine='xlsxwriter')

svmparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

svmresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

svm_roc.to_excel(writer, 
                   sheet_name='roc', 
                   index=False)

svm_auc.to_excel(writer, 
                sheet_name='auc', 
                index=False)

svmCVresult.to_excel(writer, 
                     sheet_name='CVresult', 
                     index=False)

writer.save()


#%%Logistic Regression
colNames = df.columns
colNames = colNames.drop('label')


print('Log Reg')
lrparams = {'penalty' : ['l1', 'l2'], 
            'C': np.arange(0.05, 1.0, 0.05), 
            'fit_intercept': [True, False]}

lrsearch = GridSearchCV(estimator=LogisticRegression(), 
                         param_grid=lrparams, 
                         cv=5)
print('    Fit')
lrsearch.fit(X_train, y_train)
lr_pred = lrsearch.predict(X_test)
lrAcc = accuracy_score(y_test, lr_pred)

score = lrsearch.decision_function(X_test)
score_roc = roc_curve(y_test, score, 'male')
lr_roc = pd.DataFrame({'fpr': score_roc[0], 
                       'tpr': score_roc[1]})

lr_auc = pd.DataFrame({'auc': [auc(score_roc[0], score_roc[1])]})

lrCVresult = pd.DataFrame(lrsearch.cv_results_)


#Save Results
print('    Save Results')
lrparams = hp.bestParamDF(lrsearch)
lrresult = pd.DataFrame({'true': y_test, 
                          'pred': lr_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\lr_results.xlsx', 
                        engine='xlsxwriter')

lrparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

lrresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

lr_roc.to_excel(writer, 
                   sheet_name='roc', 
                   index=False)

lr_auc.to_excel(writer, 
                   sheet_name='auc', 
                   index=False)

lrCVresult.to_excel(writer, 
                   sheet_name='CVresult', 
                   index=False)


writer.save()





#%%KNN
pcadf = PCA(n_components=9).fit(Xnorm.T).components_.T
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pcadf, 
                                                    df.label, 
                                                    test_size=0.33,
                                                    random_state=0)
#print('KNN PCA')
knnparams = {'n_neighbors': np.arange(1, 31, 2), 
             'weights': ['uniform', 'distance'], 
             'algorithm': ['auto', 'ball_tree', 
                           'kd_tree', 'brute'],
             'p': [1, 2]}

knnsearch = GridSearchCV(estimator=KNeighborsClassifier(), 
                          param_grid=knnparams, 
                          cv=5)
#print('    Fit')
knnsearch.fit(X_train_pca, y_train_pca)

knn_pred = knnsearch.predict(X_test_pca)

#Model Evaluation
knnsearch.best_score_
knnsearch.best_params_
knnAcc = accuracy_score(y_test_pca, knn_pred)
knnCM = pd.crosstab(y_test_pca, knn_pred, 
                    rownames=['True'], colnames=['Predicted'], 
                    margins=True)

knnGSresults = pd.DataFrame(knnsearch.cv_results_)

probs = knnsearch.predict_proba(X_test_pca)
male_score = []
for i in probs:
    male_score.append(i[1])
fpr, tpr, threshs = roc_curve(y_test_pca, male_score, 'male')
knn_roc = pd.DataFrame({'fpr': fpr, 
                       'tpr': tpr})

knn_auc = pd.DataFrame({'auc': [auc(fpr, tpr)]})

#Save Results
print('    Save Results')
knnparams = hp.bestParamDF(knnsearch)
knnresult = pd.DataFrame({'true': y_test_pca, 
                          'pred': knn_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\knn_pca_results.xlsx', 
                        engine='xlsxwriter')

knnparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

knnresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

knnGSresults.to_excel(writer, 
                      sheet_name='GSresult', 
                      index=False)

knn_roc.to_excel(writer, 
                      sheet_name='roc', 
                      index=False)

knn_auc.to_excel(writer, 
                sheet_name='auc', 
                index=False)

writer.save()


#%% SVM
pcadf = PCA(n_components=13).fit(df.drop('label', axis=1).T).components_.T
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pcadf, 
                                                    df.label, 
                                                    test_size=0.33,
                                                    random_state=0)

print('SVM PCA')
svmparams = {'C': np.arange(5, 50, 1)}

svmsearch = GridSearchCV(estimator=SVC(), 
                         param_grid=svmparams, 
                         cv=5)
print('    Fit')
svmsearch.fit(X_train_pca, y_train_pca)

svm_pred = svmsearch.predict(X_test_pca)

svmAcc = accuracy_score(y_test_pca, svm_pred)
svmCM = pd.crosstab(y_test_pca, svm_pred, 
                    rownames=['True'], colnames=['Predicted'], 
                    margins=True)

score = svmsearch.decision_function(X_test_pca)
score_roc = roc_curve(y_test_pca, score, 'male')
svm_roc = pd.DataFrame({'fpr': score_roc[0], 
                       'tpr': score_roc[1]})

svm_auc = pd.DataFrame({'auc': [auc(score_roc[0], score_roc[1])]})

svmCVresult = pd.DataFrame(svmsearch.cv_results_)

#Save Results
print('    Save Results')
svmparams = hp.bestParamDF(svmsearch)
svmresult = pd.DataFrame({'true': y_test_pca, 
                          'pred': svm_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\svm_pca_results.xlsx', 
                        engine='xlsxwriter')

svmparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

svmresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

svm_roc.to_excel(writer, 
                   sheet_name='roc', 
                   index=False)

svm_auc.to_excel(writer, 
                sheet_name='auc', 
                index=False)

svmCVresult.to_excel(writer, 
                     sheet_name='CVresult', 
                     index=False)
writer.save()


#%%Logistic Regression
x = df.drop('label', axis=1).values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_norm = pd.DataFrame(x_scaled, 
                      columns=colNames)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(x_norm, 
                                                    df.label, 
                                                    test_size=0.33,
                                                    random_state=0)

print('Log Reg Norm')
lrparams = {'penalty' : ['l1', 'l2'], 
            'C': np.arange(0.05, 1, 0.05), 
            'fit_intercept': [True, False]}

lrsearch = GridSearchCV(estimator=LogisticRegression(), 
                         param_grid=lrparams, 
                         cv=5)
print('    Fit')
lrsearch.fit(X_train_pca, y_train_pca)
lr_pred = lrsearch.predict(X_test_pca)
lrAcc = accuracy_score(y_test_pca, lr_pred)


score = lrsearch.decision_function(X_test_pca)
score_roc = roc_curve(y_test_pca, score, 'male')

lr_roc = pd.DataFrame({'fpr': score_roc[0], 
                       'tpr': score_roc[1]})

lr_auc = pd.DataFrame({'auc': [auc(score_roc[0], score_roc[1])]})

lrCVresult = pd.DataFrame(lrsearch.cv_results_)

#Save Results
print('    Save Results')
lrparams = hp.bestParamDF(lrsearch)
lrresult = pd.DataFrame({'true': y_test_pca, 
                          'pred': lr_pred})
resultFolder = os.getcwd() + r'\results'

writer = pd.ExcelWriter(resultFolder + r'\lr_pca_results.xlsx', 
                        engine='xlsxwriter')

lrparams.to_excel(writer, 
                   sheet_name='params', 
                   index=False)

lrresult.to_excel(writer, 
                   sheet_name='result', 
                   index=False)

lr_roc.to_excel(writer, 
                sheet_name='roc', 
                index=False)

lr_auc.to_excel(writer, 
                sheet_name='auc', 
                index=False)

lrCVresult.to_excel(writer, 
                sheet_name='CVresult', 
                index=False)

writer.save()
