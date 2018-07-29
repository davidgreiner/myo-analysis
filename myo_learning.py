import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from pomegranate import *
import matplotlib.pyplot as plt

def ten_fold_decision_tree(df_myo, train_labels, labels):
	X = df_myo[train_labels].as_matrix().astype(np.float64)
	Y = df_myo['label']
	k_fold = KFold(len(Y), shuffle=True, random_state=0)
	clf = DecisionTreeClassifier(criterion = "gini")
	y_pred = cross_val_predict(clf,X,Y,cv=k_fold, n_jobs=-1)
	print("Decision-Tree", file=open("output.txt", "a"))
	print(classification_report(Y, y_pred, target_names=labels), file=open("output.txt", "a"))
	return confusion_matrix(Y,y_pred,labels=labels)

def ten_fold_svm(df_myo, train_labels, labels):
	X = df_myo[train_labels].as_matrix().astype(np.float64)
	Y = df_myo['label']
	k_fold = KFold(len(Y), shuffle=True, random_state=0)
	clf = svm.SVC(kernel='rbf', C=1.0)
	y_pred = cross_val_predict(clf,X,Y,cv=k_fold, n_jobs=-1)
	print("SVM", file=open("output.txt", "a"))
	print(classification_report(Y, y_pred, target_names=labels), file=open("output.txt", "a"))
	return confusion_matrix(Y,y_pred,labels=labels)

def ten_fold_knn(df_myo, train_labels, kNN, labels):
	X = df_myo[train_labels].as_matrix().astype(np.float64)
	Y = df_myo['label']
	k_fold = KFold(len(Y), shuffle=True, random_state=0)
	clf = KNeighborsClassifier(n_neighbors = kNN, weights='uniform', algorithm='auto')
	y_pred = cross_val_predict(clf,X,Y,cv=k_fold, n_jobs=-1)
	print("kNN" + str(kNN), file=open("output.txt", "a"))
	print(classification_report(Y, y_pred, target_names=labels), file=open("output.txt", "a"))
	return confusion_matrix(Y,y_pred,labels=labels)

def hmm_pp(df_myo, train_labels, labels):
	print("Training HMM")
	df_grouped = df_myo.groupby(df_myo['group'])
	np.set_printoptions(suppress=True)
	cpr_X = list()
	#test_model = HiddenMarkovModel()
	#test_model.bake()
	for name, group in df_grouped:
		if group['label'].iloc[0] == 'cpr':
			cpr = group[train_labels].as_matrix()
			#test_model.fit(cpr_X, algorithm='baum-welch')
			cpr_X.append(cpr)
	cpr_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=cpr_X, algorithm='baum-welch')
	cpr_model.bake()
#plt.figure( figsize=(10,6) )
#cpr_model.plot()
	print(cpr_model.to_json())
	probability = cpr_model.log_probability(cpr)
	print(probability)

def hmm(df_myo, train_labels, labels):
	df_grouped = df_myo.groupby([df_myo['participant'],df_myo['group']])
	np.set_printoptions(suppress=True)
	cpr_X = []
	for name, group in df_grouped:
		if group['label'].iloc[0] == 'cpr':
			cpr = group[train_labels].as_matrix().transpose()
			cpr_X = np.append(cpr, axis=1)
	print(cpr_X)
	cpr_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=cpr_X)
