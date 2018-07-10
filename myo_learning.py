import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier

def ten_fold_decision_tree(df_myo, labels):
	X = df_myo[labels].as_matrix().astype(np.float64)
	Y = df_myo['label']
	k_fold = KFold(len(Y), shuffle=True, random_state=0)
	clf = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
	return cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)

def ten_fold_svm(df_myo, labels):
	X = df_myo[labels].as_matrix().astype(np.float64)
	Y = df_myo['label']
	k_fold = KFold(len(Y), shuffle=True, random_state=0)
	clf = svm.SVC(kernel='poly', degree=3, C=1.0)
	return cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)

def ten_fold_knn(df_myo, labels):
	X = df_myo[labels].as_matrix().astype(np.float64)
	Y = df_myo['label']
	k_fold = KFold(len(Y), shuffle=True, random_state=0)
	clf = KNeighborsClassifier(n_neighbors = 2, weights='uniform', algorithm='auto')
	return cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)
