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
    
     # Avoid division by zero
    if signal_std == 0:
        signal_normalized = np.zeros_like(signal)
    else:
        signal_normalized = (signal - signal_mean) / signal_std
    
    return signal_normalized
    
    # # Subtract the mean and divide by the standard deviation
    # signal_normalized = (signal - signal_mean) / signal_std
    
    # return signal_normalized
    
def truncate_signals(signal1, signal2):
    min_length = min(len(signal1), len(signal2))
    return signal1[:min_length], signal2[:min_length]
    
#'B','C','D','E','F','G'
group_name = ['A','B','C','D','E','F','G']  
weeks = ['1W','2W', '3W', '4W']
section_num = ['S1','S2'] 

# Load data from CSV files
total_path = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/total_synchrony(delta).csv'
total_synchrony = pd.DataFrame()

for group in tqdm(group_name, desc = "Groups"): # desc: 진행 바 앞에 문자열을 출력하기 위해 쓰는 키워드 
    for week in tqdm(weeks, desc=f"Weeks for group {group}"):
        for section in tqdm(section_num, desc=f"section for week {week}"):
            csv_files = [
                f'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/{group}_group_delta/Face_{week}_{group}1_{section}.csv',
                f'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/{group}_group_delta/Face_{week}_{group}2_{section}.csv',
                f'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/{group}_group_delta/Face_{week}_{group}3_{section}.csv',
                f'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/{group}_group_delta/Face_{week}_{group}4_{section}.csv'
            ]
            
            data_xlse = [] 
            vaild_files = True # 파일 안에 데이터가 있는지 없는지
            for file in csv_files:
                if os.path.exists(file): # csv_files[0]
                    df = pd.read_csv(file)
                    if df.empty:
                        print(f"File {file} is empty. Skip this set")
                        vaild_files = False
                    data_xlse.append(df)
                else:
                    print(f"File {file} does not exist. Skip this set")
                    vaild_files = False
                    break
                
            if vaild_files:
                # 길이가 맞지 않는 데이터의 경우 넘어갈 수 있도록 설정. 
                lengths = [len(df) for df in data_xlse]
                #print(lengths)
                if len(set(lengths)) != 1:
                    print(f"Files for group {group}, week {week}, section {section} have different lengths. Skipping this set.")
                    continue
                # 추출하고 싶은 데이터 값을 넣어 둠. 
                data_xlse = [pd.read_csv(file) for file in csv_files]
                data = [df['Delta_X'] for df in data_xlse] # X, Y, Z, Delta_X, Delta_Y, Delta_Z, Lip_Distance
                
                frame_rate = 25
                data_frame_section = pd.DataFrame() # 네 개의 신호 간 모든 쌍에 대해 상관 관계를 계산하여 section_means 값에 저장. 
                section_means = []
                
                for i in range(4):
                    for j in range(i+1, 4):
                        signal1 = np.array(data[i])
                        #print("signal1 : ", signal1)
                        signal2 = np.array(data[j])
                        #print("signal2 : ", signal2)
                        
                        signal1, signal2 = truncate_signals(signal1, signal2)
                        
                        signal1 = zscore_signal(signal1)
                        signal2 = zscore_signal(signal2)
                        
                        window_size = 60 * frame_rate
                        corr_data = rolling_window_correlation(signal1, signal2, window_size)
                        section_means.append(np.mean(corr_data))
                        col_name = f"{i}_{j}"
                        data_frame_section[col_name] = corr_data
                
                total_synchrony[f'{group}_{week}_{section}'] = section_means
                excel_path = f'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/Save_File_delta/Face_{week}_{group}_{section}_(delta)_Synchrony.csv'
                data_frame_section.to_csv(excel_path, index=False)

total_synchrony.to_csv(total_path, index=False)