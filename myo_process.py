#from comet_ml import Experiment

#experiment  = Experiment(api_key="JlDk1CHg96OApu8v22ihYBVqC",
#						 project_name="Masters Thesis")

from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def butter_highpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
	b, a = butter_highpass(cutoff, fs, order=order)
	y = filtfilt(b, a, data)
	return y

def process_EMG(df_myo):
	hand = df_myo[' Arm'].iloc[0]
	# High-pass filter for EMG data
	# Frequency in Hz
	highpass = 20
	# Sample rate (as collected by the MYO) in Hz
	samplerate_highpass = 200
	
	cutoff_emg_1 = butter_highpass_filter(df_myo[' EMG_1'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_1_' + hand : cutoff_emg_1.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	cutoff_emg_2 = butter_highpass_filter(df_myo[' EMG_2'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_2_' + hand : cutoff_emg_2.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	cutoff_emg_3 = butter_highpass_filter(df_myo[' EMG_3'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_3_' + hand : cutoff_emg_3.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	cutoff_emg_4 = butter_highpass_filter(df_myo[' EMG_4'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_4_' + hand : cutoff_emg_4.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	cutoff_emg_5 = butter_highpass_filter(df_myo[' EMG_5'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_5_' + hand : cutoff_emg_5.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	cutoff_emg_6 = butter_highpass_filter(df_myo[' EMG_6'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_6_' + hand : cutoff_emg_6.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	cutoff_emg_7 = butter_highpass_filter(df_myo[' EMG_7'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_7_' + hand : cutoff_emg_7.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	cutoff_emg_8 = butter_highpass_filter(df_myo[' EMG_8'].values, highpass, samplerate_highpass, 4)
	kwargs = {'Cutoff_EMG_8_' + hand : cutoff_emg_8.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	df_myo = df_myo.dropna()

	return df_myo

def process_IMU(df_myo):
	hand = df_myo[' Arm'].iloc[0]
	# 4th Butterworth band-pass filter for Accelerometer and Gyroscope data
	# Cutoff frequencies in Hz
	highcut = 0.2
	lowcut = 15
	# Sample rate (as collected by the MYO) in Hz
	samplerate_butterworth = 200
	
	# Accelerometer
	smoothed_acc_x = butter_bandpass_filter(df_myo[' Acc_X'].values, lowcut, highcut, samplerate_butterworth, 4)
	kwargs = {'Smoothed_Acc_X_' + hand : smoothed_acc_x.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	smoothed_acc_y = butter_bandpass_filter(df_myo[' Acc_Y'].values, lowcut, highcut, samplerate_butterworth, 4)
	kwargs = {'Smoothed_Acc_Y_' + hand : smoothed_acc_y.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	smoothed_acc_z = butter_bandpass_filter(df_myo[' Acc_Z'].values, lowcut, highcut, samplerate_butterworth, 4)
	kwargs = {'Smoothed_Acc_Z_' + hand : smoothed_acc_z.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	# Gyroscope
	smoothed_orientation_x = butter_bandpass_filter(df_myo[' Orientation_X'].values, lowcut, highcut, samplerate_butterworth, 4)
	kwargs = {'Smoothed_Orientation_X_' + hand : smoothed_orientation_x.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	smoothed_orientation_y = butter_bandpass_filter(df_myo[' Orientation_Y'].values, lowcut, highcut, samplerate_butterworth, 4)
	kwargs = {'Smoothed_Orientation_Y_' + hand : smoothed_orientation_y.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	smoothed_orientation_z = butter_bandpass_filter(df_myo[' Orientation_Z'].values, lowcut, highcut, samplerate_butterworth, 4)
	kwargs = {'Smoothed_Orientation_Z_' + hand : smoothed_orientation_z.tolist()}
	df_myo = df_myo.assign(**kwargs)
	
	return df_myo

if __name__ == "__main__":
	import numpy as np
	import pandas as pd
	#df = pd.read_csv('day2emgclean.csv', dtype={'Warm':'category', 'Sync':'category', 'Locked':'category', 'Pose':'category'})
	df = pd.read_hdf('day2emgclean.h5')
	
	# Remove unecessary column
	del df['DeviceID']
	
	# Split data by sensor
	dfl = df[df['Arm'] == "left"]
	dfr = df[df['Arm'] == "right"]
	
	del dfl['Arm']
	del dfr['Arm']
	
	# Data Processing
	
	dfl = process_IMU(dfl)
	
	dfl = process_EMG(dfl)


	# Save Data
	dfl.to_hdf('leftcompresssed.h5', 'data', mode='w', format='table', complevel=3)
	dfr.to_hdf('rightcompresssed.h5', 'data', mode='w', format='table', complevel=3)
