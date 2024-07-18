import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import butter, filtfilt

# 로우패스 필터 설계 함수
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Z-score normalization 함수
def zscore_signal(signal):
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std == 0:
        signal_normalized = np.zeros_like(signal)
    else:
        signal_normalized = (signal - signal_mean) / signal_std
    return signal_normalized

# 성과 총점 읽어오기
performance_file = 'C:/Users/user/Desktop/Group_performance.xlsx'  # 성과 총점이 기록된 엑셀 파일 경로
performance_df = pd.read_excel(performance_file)
performance_scores = performance_df['A']  # 그룹 별 성과 총점이 기록된 열 이름

# CSV 파일 목록
csv_files = [
    'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/A_group/Face_1W_A1_S1.csv',
    'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/A_group/Face_1W_A2_S1.csv',
    'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/A_group/Face_1W_A3_S1.csv',
    'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/A_group/Face_1W_A4_S1.csv',
]

# CSV 파일에서 'X' 칼럼 값 읽어오기
data_x = []
valid_files = True
for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        if df.empty:
            print(f"File {file} is empty. Skipping this set.")
            valid_files = False
            break
        avg_x = df['X'].mean()
        data_x.append(avg_x)
    else:
        print(f"File {file} does not exist. Skipping this set.")
        valid_files = False
        break

if valid_files:
    if len(data_x) != len(performance_scores):
        print("The number of groups in the performance file does not match the number of CSV files.")
    else:
        # 상관관계 계산
        correlation_with_performance = np.corrcoef(data_x, performance_scores)[0, 1]

        # 산점도 그리기
        plt.figure(figsize=(10, 6))
        plt.scatter(data_x, performance_scores, label=f'Correlation: {correlation_with_performance:.2f}')
        
        plt.xlabel('Average X Values')
        plt.ylabel('Performance Scores')
        plt.title('Correlation between Average X Values and Performance Scores')
        plt.legend()
        plt.grid(True)
        plt.show()
else:
    print("Valid data files not found. Processing skipped")