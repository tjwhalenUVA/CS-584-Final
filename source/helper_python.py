# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:12:45 2017

@author: timothy.whalen
"""
from sklearn import tree
import pydotplus
import os
import pandas as pd

def dtDFGenerator(tree, X_train):
    import pandas as pd
    tru = []
    fls = []
    for t in tree.value:
        for i in t:
            tru.append(i[0])
            fls.append(i[1])
    
    df = pd.DataFrame({'left':tree.children_left, 
                       'right':tree.children_right, 
                       'feature':tree.feature, 
                       'threshold':tree.threshold, 
                       'impurity':tree.impurity, 
                       'true':tru, 
                       'false':fls, 
                       'samples':tree.weighted_n_node_samples})
    #Get Feature Names
    feats = []
    for f in df.feature:
        if f == -2:
            feats.append('_leaf')
        else:
            feats.append(X_train.columns[f])
    df['feature_name'] = pd.Series(feats, index=df.index)
    
    #Get Feature Importance
    featImp = []
    for f in df.feature:
        if f == -2:
            featImp.append(None)
        else:
            featImp.append(tree.compute_feature_importances()[f])
    df['feature_importance'] = pd.Series(featImp, index=df.index)
    
    #Right Child
    rc = []
    for chi in df.right:
        if chi == -1:
            rc.append('_leaf')
        else:
            rc.append(df.get_value(chi, 'feature_name'))
    df['right_child'] = pd.Series(rc, index=df.index)
    
    #Left Child
    lc = []
    for chi in df.left:
        if chi == -1:
            lc.append('_leaf')
        else:
            lc.append(df.get_value(chi, 'feature_name'))
    df['left_child'] = pd.Series(lc, index=df.index)
    
    
    df = df[['feature', 'feature_name', 'feature_importance', 
             'impurity', 'samples', 
             'left_child', 'right_child', 
             'true', 'false', 'threshold']]
    return(df)

def dtplot(clf, fileName, X_train):
    dot_data = tree.export_graphviz(clf,
                                    feature_names=X_train.columns,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.getcwd() + '\Trees\%s.pdf' % fileName)


def bestParamDF(clf):
    df = pd.DataFrame(columns=['Parameter', 'Value'])
    i=0
    for k in clf.best_params_.keys():
        df.set_value(i, 'Parameter', k)
        df.set_value(i, 'Value', clf.best_params_[k])
        i += 1
    return(df)