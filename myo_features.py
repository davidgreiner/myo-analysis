import numpy as np
import pandas as pd

# Signal Magnitude Area function
def sma(df_myo):
	print(df_myo)
	return df_myo

# Root Mean Squared function
#def rms(df_myo):
#	print("RMS")
#	print(df_myo)

def calculate_EMG_features(df_myo):
	df_myo['helper1'] = df_myo['label'].shift(1)
	df_myo['helper2'] = df_myo['label'] != df_myo['helper1']
	df_myo['group'] = df_myo['helper2'].cumsum()
	
	hand = df_myo[' Arm'].iloc[0]
	df_myo['EMG_1_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_1_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['EMG_2_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_2_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['EMG_3_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_3_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['EMG_4_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_4_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['EMG_5_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_5_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['EMG_6_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_6_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['EMG_7_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_7_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['EMG_8_mean_' + hand] = df_myo.groupby(df_myo['group'])['Cutoff_EMG_8_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	
	# Root Mean Squared
	df_myo['EMG_rms_' + hand] = np.sqrt(1 / 8 * (df_myo['Cutoff_EMG_1_' + hand] ** 2 + df_myo['Cutoff_EMG_2_' + hand] ** 2 + df_myo['Cutoff_EMG_3_' + hand] ** 2 + df_myo['Cutoff_EMG_4_' + hand] ** 2 + df_myo['Cutoff_EMG_5_' + hand] ** 2 + df_myo['Cutoff_EMG_6_' + hand] ** 2 + df_myo['Cutoff_EMG_7_' + hand] ** 2 + df_myo['Cutoff_EMG_8_' + hand] ** 2))
	df_myo['EMG_rms_' + hand] = df_myo.groupby(df_myo['group'])['EMG_rms_' + hand].rolling('2s').mean().reset_index(0,drop=True)

	# Signal Magnitude Area
	df_myo['EMG_sma_' + hand] = df_myo['Cutoff_EMG_1_' + hand] + df_myo['Cutoff_EMG_2_' + hand] + df_myo['Cutoff_EMG_3_' + hand] + df_myo['Cutoff_EMG_4_' + hand] + df_myo['Cutoff_EMG_5_' + hand] + df_myo['Cutoff_EMG_6_' + hand] + df_myo['Cutoff_EMG_7_' + hand] + df_myo['Cutoff_EMG_8_' + hand]
	df_myo['EMG_sma_' + hand] = df_myo.groupby(df_myo['group'])['EMG_sma_' + hand].rolling('2s').sum().reset_index(0,drop=True)
	
	del df_myo['helper1']
	del df_myo['helper2']
	del df_myo['group']

	return df_myo

def calculate_IMU_features(df_myo):
	df_myo['helper1'] = df_myo['label'].shift(1)
	df_myo['helper2'] = df_myo['label'] != df_myo['helper1']
	df_myo['group'] = df_myo['helper2'].cumsum()
	
	hand = df_myo[' Arm'].iloc[0]
	# Mean
	df_myo['Acc_X_mean_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Acc_X_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['Acc_Y_mean_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Acc_Y_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['Acc_Z_mean_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Acc_Z_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	
	df_myo['Orientation_X_mean_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Orientation_X_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['Orientation_Y_mean_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Orientation_Y_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	df_myo['Orientation_Z_mean_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Orientation_Z_' + hand].rolling('2s').mean().reset_index(0,drop=True)
	
	# Standard Deviation
	df_myo['Acc_X_std_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Acc_X_' + hand].rolling('2s').std().reset_index(0,drop=True)
	df_myo['Acc_Y_std_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Acc_Y_' + hand].rolling('2s').std().reset_index(0,drop=True)
	df_myo['Acc_Z_std_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Acc_Z_' + hand].rolling('2s').std().reset_index(0,drop=True)
	
	df_myo['Orientation_X_std_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Orientation_X_' + hand].rolling('2s').std().reset_index(0,drop=True)
	df_myo['Orientation_Y_std_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Orientation_Y_' + hand].rolling('2s').std().reset_index(0,drop=True)
	df_myo['Orientation_Z_std_' + hand] = df_myo.groupby(df_myo['group'])['Smoothed_Orientation_Z_' + hand].rolling('2s').std().reset_index(0,drop=True)
	
	# Signal Magnitude Area
	df_myo['Acc_sma_' + hand] = df_myo['Smoothed_Acc_X_' + hand] + df_myo['Smoothed_Acc_Y_' + hand] + df_myo['Smoothed_Acc_Z_' + hand]
	df_myo['Acc_sma_' + hand] = df_myo.groupby(df_myo['group'])['Acc_sma_' + hand].rolling('2s').sum().reset_index(0,drop=True)
	
	df_myo['Orientation_sma_' + hand] = df_myo['Smoothed_Orientation_X_' + hand] + df_myo['Smoothed_Orientation_Y_' + hand] + df_myo['Smoothed_Orientation_Z_' + hand]
	df_myo['Orientation_sma_' + hand] = df_myo.groupby(df_myo['group'])['Orientation_sma_' + hand].rolling('2s').sum().reset_index(0,drop=True)

	del df_myo['helper1']
	del df_myo['helper2']
	del df_myo['group']

	return df_myo

if __name__ == "__main__":
	print("Please call functions seperatly")
