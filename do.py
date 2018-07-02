import myo_analysis as ma
import myo_process as mp
import myo_features as mf
import myo_learning as ml
import pandas as pd

try:
	df_merged = pd.read_pickle("data.pkl")
except:
	df_merged = pd.DataFrame()

	participants = ['5187','7073','4501','1571','2334','5557','2636','2170','4623']

	for participant in participants:
		for day in ['1','2','3']:
			try:
				df = pd.read_pickle(participant + "_Day_" + day + "_EMG_labeled.pkl")
			except:
				df = ma.import_myo_csv_file("", participant + "_Day_" + day + "_EMG.csv")
				df = ma.label_data(df_emg,"", participant + "_Day" + day + ".txt")
				df.to_pickle(participant + "_Day_" + day + "_EMG_labeled.pkl")

			df_l, df_r = ma.parse_myo_hand(df)

			df_l = mp.process_EMG(df_l)
			df_r = mp.process_EMG(df_r)

			df_l = mp.process_IMU(df_l)
			df_r = mp.process_IMU(df_r)
			
			pd.set_option('display.max_columns', None)
			
			df_l = df_l.set_index(' Timestamp')
			df_r = df_r.set_index(' Timestamp')

			df_l = mf.calculate_EMG_features(df_l)
			df_r = mf.calculate_EMG_features(df_r)

			df_l = mf.calculate_IMU_features(df_l)
			df_r = mf.calculate_IMU_features(df_r)

			df_l = df_l.dropna()
			df_r = df_r.dropna()

			df_l = df_l[::200]
			df_r = df_r[::200]

			columns = ['Device ID', ' Warm?', ' Sync', 	' Orientation_W', ' Orientation_X', ' Orientation_Y', ' Orientation_Z', ' Acc_X', ' Acc_Y', ' Acc_Z', ' Gyro_X', ' Gyro_Y', ' Gyro_Z', ' Pose', ' EMG_1', ' EMG_2', ' EMG_3', ' EMG_4', ' EMG_5', ' EMG_6', ' EMG_7', ' EMG_8', 'Locked', ' RSSI', ' Roll', ' Pitch', ' Yaw ']
			
			df_l = df_l.rename(columns=lambda x: x+'_left' if x in columns else x)
			df_r = df_r.rename(columns=lambda x: x+'_right' if x in columns else x)
			
			df_l = df_l.drop(' Arm', 1)
			df_r = df_r.drop(' Arm', 1)

			df_l = df_l.reset_index()
			df_r = df_r.reset_index()

			df_l[' Timestamp'] = df_l[' Timestamp'].dt.round('1s')
			df_r[' Timestamp'] = df_r[' Timestamp'].dt.round('1s')
			
			df = df_l.merge(df_r, on=[' Timestamp', 'label'], how='inner')
			#print(df)
			
			df_merged = df_merged.append(df)
	df_merged.to_pickle("data.pkl")



#score_dt = ml.ten_fold_decision_tree(df_merged, ['EMG_1_mean','EMG_2_mean','EMG_3_mean','EMG_4_mean','EMG_5_mean','EMG_6_mean','EMG_7_mean','EMG_8_mean'])

#print(score, file=open("output.txt", "a"))
