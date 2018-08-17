import myo_analysis as ma
import myo_process as mp
import myo_features as mf
import myo_learning as ml
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

train_columns = ['EMG_1_mean_left','EMG_2_mean_left','EMG_3_mean_left','EMG_4_mean_left','EMG_5_mean_left','EMG_6_mean_left','EMG_7_mean_left','EMG_8_mean_left','EMG_rms_left','EMG_sma_left','Acc_X_mean_left','Acc_Y_mean_left','Acc_Z_mean_left','Orientation_X_mean_left','Orientation_Y_mean_left','Orientation_Z_mean_left','Acc_X_std_left','Acc_Y_std_left','Acc_Z_std_left','Orientation_X_std_left','Orientation_Y_std_left','Orientation_Z_std_left','Acc_sma_left','Orientation_sma_left','EMG_1_mean_right','EMG_2_mean_right','EMG_3_mean_right','EMG_4_mean_right','EMG_5_mean_right','EMG_6_mean_right','EMG_7_mean_right','EMG_8_mean_right','EMG_rms_right','EMG_sma_right','Acc_X_mean_right','Acc_Y_mean_right','Acc_Z_mean_right','Orientation_X_mean_right','Orientation_Y_mean_right','Orientation_Z_mean_right','Acc_X_std_right','Acc_Y_std_right','Acc_Z_std_right','Orientation_X_std_right','Orientation_Y_std_right','Orientation_Z_std_right','Acc_sma_right','Orientation_sma_right']

participants = ['2253','5187','7073','4501','2334','5557','2636','2170','4623'] #['2253','5187','7073','4501','1571','2334','5557','2636','2170','4623']

window_sizes = ['6s'] #['2s','4s','6s']

for window_size in window_sizes:
	#print(window_size, file=open("output.txt", "a"))
	try:
		df_merged = pd.read_pickle("data_" + window_size + ".pkl")
	except:
		df_merged = pd.DataFrame()

		for participant in participants:
			for day in ['1','2','3']:
				try:
					df = pd.read_pickle(participant + "_Day_" + day + "_EMG_labeled.pkl")
				except:
					df = ma.import_myo_csv_file("", participant + "_Day_" + day + "_EMG.csv")
					df = ma.label_data(df,"", participant + "_Day" + day + ".txt")
					df.to_pickle(participant + "_Day_" + day + "_EMG_labeled.pkl")
				
				df['participant'] = participant
				df['day'] = day

				df_l, df_r = ma.parse_myo_hand(df)

				df_l = mp.process_EMG(df_l)
				df_r = mp.process_EMG(df_r)

				df_l = mp.process_IMU(df_l)
				df_r = mp.process_IMU(df_r)
				
				pd.set_option('display.max_columns', None)
				
				df_l = df_l.set_index(' Timestamp')
				df_r = df_r.set_index(' Timestamp')

				df_l['helper1'] = df_l['label'].shift(1)
				df_l['helper2'] = df_l['label'] != df_l['helper1']
				df_l['group'] = df_l['helper2'].cumsum()
				del df_l['helper1']
				del df_l['helper2']
				
				df_r['helper1'] = df_r['label'].shift(1)
				df_r['helper2'] = df_r['label'] != df_r['helper1']
				df_r['group'] = df_r['helper2'].cumsum()
				del df_r['helper1']
				del df_r['helper2']

				df_l = mf.calculate_EMG_features(df_l, window_size)
				df_r = mf.calculate_EMG_features(df_r, window_size)

				df_l = mf.calculate_IMU_features(df_l, window_size)
				df_r = mf.calculate_IMU_features(df_r, window_size)
				
				if day == '3':
					print(participant, file=open("average.txt", "a"))
					mf.calculate_average_time(df_l)
				
				df_l = df_l.iloc[::200, :]
				df_r = df_r.iloc[::200, :]

				df_l = df_l.dropna()
				df_r = df_r.dropna()

				columns = ['Device ID', ' Warm?', ' Sync', 	' Orientation_W', ' Orientation_X', ' Orientation_Y', ' Orientation_Z', ' Acc_X', ' Acc_Y', ' Acc_Z', ' Gyro_X', ' Gyro_Y', ' Gyro_Z', ' Pose', ' EMG_1', ' EMG_2', ' EMG_3', ' EMG_4', ' EMG_5', ' EMG_6', ' EMG_7', ' EMG_8', 'Locked', ' RSSI', ' Roll', ' Pitch', ' Yaw ']
				
				df_l = df_l.rename(columns=lambda x: x+'_left' if x in columns else x)
				df_r = df_r.rename(columns=lambda x: x+'_right' if x in columns else x)
				
				df_l = df_l.drop(' Arm', 1)
				df_r = df_r.drop(' Arm', 1)

				df_l = df_l.reset_index()
				df_r = df_r.reset_index()

				df_l[' Timestamp'] = df_l[' Timestamp'].dt.round('1s')
				df_r[' Timestamp'] = df_r[' Timestamp'].dt.round('1s')
				
				df = df_l.merge(df_r, on=[' Timestamp', 'label', 'day', 'participant','group'], how='inner')
				
				df_merged = df_merged.append(df)
		df_merged.to_pickle("data_" + window_size + ".pkl")
		df_merged.to_csv("data_" + window_size + ".csv")

	df_merged = df_merged.loc[df_merged['day'] == "3"]
	df_hmm = df_merged
	df_merged = df_merged[df_merged.label.isin(['t','w','b','o','cpr'])]

	np.set_printoptions(threshold=100)

	labels = df_merged.label.unique()
	'''
	for participant in participants:
		print(participant, file=open("output.txt", "a"))
		df_merged_p = df_merged.loc[df_merged['participant'] == participant]
		score_hmm = ml.hmm_pp(df_merged_p, train_columns, labels)
		#print(score_hmm, file=open("output.txt", "a"))
		df_cm = pd.DataFrame(score_hmm, index = labels, columns = labels)
		plt.figure(figsize = (10,7))
		sn.heatmap(df_cm, annot=True, fmt="d")
		plt.savefig("hmm_" + window_size + "_" + str(participant) + ".pdf", dpi=150)
		
		plt.clf()

		score_dt = ml.ten_fold_decision_tree(df_merged_p, train_columns, labels)
		#print(score_dt, file=open("output.txt", "a"))
		df_cm = pd.DataFrame(score_dt, index = labels, columns = labels)
		plt.figure(figsize = (10,7))
		sn.heatmap(df_cm, annot=True, fmt="d")
		plt.savefig("decision_tree_" + window_size + "_" + str(participant) + ".pdf", dpi=150)

		plt.clf()

		for kNN in range(1,6):
			score_knn = ml.ten_fold_knn(df_merged_p, train_columns, kNN, labels)
			#print(score_knn, file=open("output.txt", "a"))
			df_cm = pd.DataFrame(score_knn, index = labels, columns = labels)
			sn.heatmap(df_cm, annot=True, fmt="d")
			plt.savefig("knn" + str(kNN) + "_" + window_size + "_" + str(participant) + ".pdf", dpi=150)
			
			plt.clf()

		score_svm = ml.ten_fold_svm(df_merged_p, train_columns, labels)
		#print(score_svm, file=open("output.txt", "a"))
		df_cm = pd.DataFrame(score_svm, index = labels, columns = labels)
		sn.heatmap(df_cm, annot=True, fmt="d")
		plt.savefig("svm_" + window_size + "_" + str(participant) + ".pdf", dpi=150)

		plt.clf()
	'''
	score_hmm = ml.hmm(df_merged, train_columns, labels)
	#print(score_hmm, file=open("output.txt", "a"))
	df_cm = pd.DataFrame(score_hmm, index = labels, columns = labels)
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True, fmt="d")
	plt.savefig("hmm_" + window_size + "_without.pdf", dpi=150)

	plt.clf()
	
	score_dt = ml.ten_fold_decision_tree(df_merged, train_columns, labels)
	#print(score_dt, file=open("output.txt", "a"))
	df_cm = pd.DataFrame(score_dt, index = labels, columns = labels)
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True, fmt="d")
	plt.savefig("decision_tree_" + window_size + "_without.pdf", dpi=150)

	plt.clf()

	for kNN in range(1,6):
		score_knn = ml.ten_fold_knn(df_merged, train_columns, kNN, labels)
		#print(score_knn, file=open("output.txt", "a"))
		df_cm = pd.DataFrame(score_knn, index = labels, columns = labels)
		sn.heatmap(df_cm, annot=True, fmt="d")
		plt.savefig("knn" + str(kNN) + "_" + window_size + "_without.pdf", dpi=150)

		plt.clf()

	score_svm = ml.ten_fold_svm(df_merged, train_columns, labels)
	#print(score_svm, file=open("output.txt", "a"))
	df_cm = pd.DataFrame(score_svm, index = labels, columns = labels)
	sn.heatmap(df_cm, annot=True, fmt="d")
	plt.savefig("svm_" + window_size + "_without.pdf", dpi=150)

	plt.clf()
