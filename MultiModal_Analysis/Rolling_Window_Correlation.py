import numpy as np
import pandas as pd
import os
from tqdm import tqdm

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

def zscore_signal(signal):
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    
    # Subtract the mean and divide by the standard deviation
    signal_normalized = (signal - signal_mean) / signal_std
    
    return signal_normalized

group_name = ['A']
weeks = ['1W', '2W', '3W', '4W']
section_num = ['S1', 'S2']

# Load data from CSV files
total_path = 'D:/MultiModal/MultiModal_Model/results/face/face_synchrony/total_synchrony.csv'
total_synchrony = pd.DataFrame()

for group in tqdm(group_name, desc = "Groups"): # desc: 진행 바 앞에 문자열을 출력하기 위해 쓰는 키워드 
    for week in tqdm(weeks, desc=f"Weeks for group {group}"):
        for section in tqdm(section_num, desc=f"section for week {week}"):
            csv_files = [
                f'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/{group}/Face_{week}_{group}1_{section}.csv',
                f'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/{group}/Face_{week}_{group}2_{section}.csv',
                f'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/{group}/Face_{week}_{group}3_{section}.csv',
                f'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/{group}/Face_{week}_{group}4_{section}.csv'
            ]
            if os.path.exists(csv_files[0]):
                # Load data and extract the column of interest
                data_xlse = [pd.read_csv(file) for file in csv_files]
                data = [df['X'] for df in data_xlse] # X, Y, Z, Delta_X, Delta_Y, Delta_Z, Lip_Distance
                
                frame_rate = 25
                data_frame_section = pd.DataFrame()
                section_means = []
                
                for i in range(4):
                    for j in range(i+1, 4):
                        signal1 = np.array(data[i])
                        signal2 = np.array(data[j])
                        
                        signal1 = zscore_signal(signal1)
                        signal2 = zscore_signal(signal2)
                        
                        window_size = 60 * frame_rate
                        corr_data = rolling_window_correlation(signal1, signal2, window_size)
                        section_means.append(np.mean(corr_data))
                        col_name = f"{i}_{j}"
                        data_frame_section[col_name] = corr_data
                
                total_synchrony[f'{group}_{week}_{section}'] = section_means
                excel_path = f'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Data/{group}/Face_{week}_{section}_Synchrony.csv'
                data_frame_section.to_csv(excel_path, index=False)

total_synchrony.to_csv(total_path, index=False)