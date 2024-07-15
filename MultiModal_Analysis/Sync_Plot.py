import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt

def directional_agreement(signal1, signal2):
    # Normalize the signals
    signal1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
    
    # Compute the dot product
    dot_product = np.dot(signal1_norm, signal2_norm)
    
    # Compute the directional agreement
    directional_agreement = dot_product / len(signal1)
    
    return directional_agreement

# 로우패스 필터 설계
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def cross_correlation(signal1, signal2):
    # Time lag range (-len(signal1)+1, len(signal1)-1)
    time_lags = np.arange(-len(signal1) + 1, len(signal1))
    # Compute cross-correlation for each time lag
    cross_corr = [np.correlate(signal1, np.roll(signal2, shift), mode='valid')[0] for shift in time_lags]
    
    return np.array(cross_corr), time_lags


def pearson_cross_correlation(signal1, signal2):
    # Subtract the mean
    signal1_mean = np.mean(signal1)
    signal2_mean = np.mean(signal2)
    
    signal1_adjusted = signal1 - signal1_mean
    signal2_adjusted = signal2 - signal2_mean
    
    # Calculate Pearson correlation coefficient
    numerator = np.sum(signal1_adjusted * signal2_adjusted)
    denominator = np.sqrt(np.sum(signal1_adjusted ** 2) * np.sum(signal2_adjusted ** 2))
    
    pearson_corr = numerator / denominator
    
    return pearson_corr


def rolling_window_correlation(signal1, signal2, window_size):
    num_samples = len(signal1)
    correlations = []

    for i in range(num_samples - window_size + 1):
        window1 = signal1[i : i + window_size]
        window2 = signal2[i : i + window_size]
        
        # Calculate Pearson correlation coefficient for the current window
        correlation = np.corrcoef(window1, window2)[0, 1]
        correlations.append(correlation)

    return np.array(correlations)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def zscore_signal(signal):
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    
    # Subtract the mean and divide by the standard deviation
    signal_normalized = (signal - signal_mean) / signal_std
    
    return signal_normalized


# Load data from CSV files
csv_files = [
    'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/A/Face_1W_A1_S1.csv',
    'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/A/Face_1W_A2_S1.csv',
    'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/A/Face_1W_A3_S1.csv',
    'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/A/Face_1W_A4_S1.csv'
]


'''
csv_files = [
    'D:/Drive_Files/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/head_pose_coordinates_delta_A1.csv',
    'D:/Drive_Files/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/head_pose_coordinates_delta_A2.csv',
    'D:/Drive_Files/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/head_pose_coordinates_delta_A3.csv',
    'D:/Drive_Files/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/head_pose_coordinates_delta_A4.csv'
]
'''

# Load data and extract the column of interest
data_xlse = [pd.read_csv(file) for file in csv_files]
data = [df['X'] for df in data_xlse] # X , Y, Z, Delta_X, Delta_Y, Delta_Z

# Define frame rate and time window
frame_rate = 25  # frames per second
start_time = 0  # in seconds
end_time =  1199 # in seconds

start_frame = start_time * frame_rate
end_frame = end_time * frame_rate

#start_frame = 0
#end_frame = 1200

# 샘플 신호 생성
fs = frame_rate  # 샘플링 주파수 (Hz)
num_samples = end_frame - start_frame
t = np.linspace(start_time, end_time, num_samples, endpoint=False)


# 로우패스 필터 적용
cutoff = 0.5  # 커트오프 주파수 (Hz)

'''
# Specify the bandpass filter parameters
lowcut = 0.1  # Low cutoff frequency in Hz
highcut = 0.5  # High cutoff frequency in Hz
fs = 25  # Sampling frequency in Hz
order = 5  # Filter order
'''

signal1_raw = data[0][start_frame:end_frame].to_numpy()
signal2_raw = data[1][start_frame:end_frame].to_numpy()

signal1_raw = zscore_signal(signal1_raw)
signal2_raw = zscore_signal(signal2_raw)

# Extract data within the specified window
signal1 = lowpass_filter(signal1_raw, cutoff, fs)
signal2 = lowpass_filter(signal2_raw, cutoff, fs)

signal1 = zscore_signal(signal1)
signal2 = zscore_signal(signal2)

# Directional Agreement를 계산합니다.
#da_score = directional_agreement(signal1, signal2)


# Cross Correlation 계산
#cross_corr_values, time_lags = cross_correlation(signal1, signal2)

# Calculate Pearson correlation coefficients for raw and filtered signals at each time lag
#cross_corr_values_raw, time_lags_raw = cross_correlation(signal1_raw, signal2_raw)
#cross_corr_values_filtered, time_lags_filtered = cross_correlation(signal1, signal2)

# Calculate rolling window correlation
window_size = 60 * frame_rate
rolling_correlations_raw = rolling_window_correlation(signal1_raw, signal2_raw, window_size)
rolling_correlations_filtered = rolling_window_correlation(signal1, signal2, window_size)


# Generate the time index for rolling correlations
rolling_time_index = np.arange(len(rolling_correlations_raw))

# Convert frame indices to time in seconds
time_seconds = np.linspace(start_time, end_time, len(signal1_raw))

# Plot additional Rolling Window Correlation alongside existing plots
plt.figure(figsize=(12, 10))

# Plot 1: Signals (Raw and Filtered)
plt.subplot(3, 1, 1)
plt.plot(time_seconds,signal1_raw, label='Signal 1 Raw', color='blue')
plt.plot(time_seconds,signal2_raw, label='Signal 2 Raw', color='orange')
plt.plot(time_seconds,signal1, label='Signal 1 Filtered', color='green')
plt.plot(time_seconds,signal2, label='Signal 2 Filtered', color='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Raw and Filtered Signals')
plt.legend()


avg_rolling_correlation_raw = np.mean(rolling_correlations_raw)
avg_rolling_correlation_filtered = np.mean(rolling_correlations_filtered)


# Plot 2: Rolling Window Correlation with average values in the title
plt.subplot(3, 1, 2)
plt.plot(time_seconds[window_size // 2:len(rolling_correlations_raw) + window_size // 2], rolling_correlations_raw, label='Rolling Raw', color='blue')
plt.plot(time_seconds[window_size // 2:len(rolling_correlations_filtered) + window_size // 2], rolling_correlations_filtered, label='Rolling Filtered', color='green')
plt.xlabel('Time (seconds)')
plt.ylabel('Rolling Window Pearson Correlation')
plt.ylim(-1.0, 1.0)
plt.title(f'Rolling Window Pearson Correlation\nAvg Raw Correlation: {avg_rolling_correlation_raw:.2f}\nAvg Filtered Correlation: {avg_rolling_correlation_filtered:.2f}')
max_time = max(time_seconds)
x_ticks = range(0, int(max_time) + 60, 60)
plt.xticks(x_ticks)
plt.legend()

plt.tight_layout()
plt.show()