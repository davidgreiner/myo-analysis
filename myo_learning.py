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
	clf = svm.SVC(kernel='rbf', C=1000.0, gamma=0.0001)
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
	df_grouped = df_myo.groupby(df_myo['group'])
	np.set_printoptions(suppress=True)
	groups = len(df_grouped)
	k_fold = KFold(groups, shuffle=True, random_state=0)
	group_name = list(df_grouped.groups.keys())
	expected = []
	predicted = []
	for train_index, test_index in k_fold.split(df_grouped):
		cpr_X = list()
		w_X = list()
		o_X = list()
		t_X = list()
		b_X = list()
		for index in train_index:
			df_train = df_grouped.get_group(group_name[index])
			if df_train['label'].iloc[0] == 'cpr':
				cpr = df_train[train_labels].as_matrix()
				cpr_X.append(cpr)
			if df_train['label'].iloc[0] == 'w':
				w = df_train[train_labels].as_matrix()
				w_X.append(w)
			if df_train['label'].iloc[0] == 'o':
				o = df_train[train_labels].as_matrix()
				o_X.append(o)
			if df_train['label'].iloc[0] == 't':
				t = df_train[train_labels].as_matrix()
				t_X.append(t)
			if df_train['label'].iloc[0] == 'b':
				b = df_train[train_labels].as_matrix()
				b_X.append(b)
		df_test = df_grouped.get_group(group_name[test_index[0]])
		test_label = df_test['label'].iloc[0]
		test = df_test[train_labels].as_matrix()
		cpr_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=cpr_X, algorithm='baum-welch')
		cpr_model.bake()
		w_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=w_X, algorithm='baum-welch')
		w_model.bake()
		o_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=o_X, algorithm='baum-welch')
		o_model.bake()
		t_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=t_X, algorithm='baum-welch')
		t_model.bake()
		b_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=b_X, algorithm='baum-welch')
		b_model.bake()
		cpr_probability = cpr_model.log_probability(test)
		w_probability = w_model.log_probability(test)
		o_probability = o_model.log_probability(test)
		t_probability = t_model.log_probability(test)
		b_probability = b_model.log_probability(test)
		probabilities = [(cpr_probability,'cpr'),(w_probability,'w'),(o_probability,'o'),(t_probability,'t'),(b_probability,'b')]
		#print(test_label)
		#print(max(probabilities))
		expected.append(test_label)
		predicted.append(max(probabilities)[1])
	print("HMM", file=open("output.txt", "a"))
	print(classification_report(expected, predicted, target_names=labels), file=open("output.txt", "a"))
	return confusion_matrix(expected,predicted,labels=labels)

def hmm(df_myo, train_labels, labels):
	df_grouped = df_myo.groupby([df_myo['participant'],df_myo['group']])
	np.set_printoptions(suppress=True)
	groups = len(df_grouped)
	k_fold = KFold(groups, shuffle=True, random_state=0)
	group_name = list(df_grouped.groups.keys())
	expected = []
	predicted = []
	for train_index, test_index in k_fold.split(df_grouped):
		print("TRAIN:", train_index, "TEST:", test_index)
		cpr_X = list()
		w_X = list()
		o_X = list()
		t_X = list()
		b_X = list()
		for index in train_index:
			df_train = df_grouped.get_group(group_name[index])
			if df_train['label'].iloc[0] == 'cpr':
				cpr = df_train[train_labels].as_matrix()
				cpr_X.append(cpr)
			if df_train['label'].iloc[0] == 'w':
				w = df_train[train_labels].as_matrix()
				w_X.append(w)
			if df_train['label'].iloc[0] == 'o':
				o = df_train[train_labels].as_matrix()
				o_X.append(o)
			if df_train['label'].iloc[0] == 't':
				t = df_train[train_labels].as_matrix()
				t_X.append(t)
			if df_train['label'].iloc[0] == 'b':
				b = df_train[train_labels].as_matrix()
				b_X.append(b)
		df_test = df_grouped.get_group(group_name[test_index[0]])
		test_label = df_test['label'].iloc[0]
		test = df_test[train_labels].as_matrix()
		cpr_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=cpr_X, algorithm='baum-welch')
		cpr_model.bake()
		w_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=w_X, algorithm='baum-welch')
		w_model.bake()
		o_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=o_X, algorithm='baum-welch')
		o_model.bake()
		t_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=t_X, algorithm='baum-welch')
		t_model.bake()
		b_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=b_X, algorithm='baum-welch')
		b_model.bake()
		cpr_probability = cpr_model.log_probability(test)
		w_probability = w_model.log_probability(test)
		o_probability = o_model.log_probability(test)
		t_probability = t_model.log_probability(test)
		b_probability = b_model.log_probability(test)
		probabilities = [(cpr_probability,'cpr'),(w_probability,'w'),(o_probability,'o'),(t_probability,'t'),(b_probability,'b')]
		#print(test_label)
		#print(max(probabilities))
		expected.append(test_label)
		predicted.append(max(probabilities)[1])
	print("HMM", file=open("output.txt", "a"))
	print(classification_report(expected, predicted, target_names=labels), file=open("output.txt", "a"))
	return confusion_matrix(expected,predicted,labels=labels)
