import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr

# 성과 평균 점수를 기록한 부분의 column 읽어오기
performance_file = 'C:/Users/user/Desktop/Group_performance.xlsx'  # 성과 총점이 기록된 엑셀 파일 경로
performance_df = pd.read_excel(performance_file)
performance_scores = performance_df.loc[0]  # 첫 번째 행의 값을 성과 점수로 설정

# CSV 파일 목록
Weeks = ['1W','2W','3W','4W']
Steps = ['S1','S2']
groups = ['A','B','C','D','E','F','G']

path = f'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/'

group_means = {group: [] for group in groups}

for group in groups:
    for week in Weeks:
        for i in ['1','2','3','4']:
            for step in Steps:
                # 디렉토리 경로 설정
                directory_path = os.path.join(path, f'{group}_group/')
                
                # 디렉토리에서 파일 목록 가져오기
                file_list = os.listdir(directory_path)
                
                for file_name in file_list:
                    # 파일 이름에 특정 주, 그룹, 단계가 포함되어 있는지 확인
                    if f'Face_{week}_{group}{i}_{step}' in file_name:
                        # 파일 경로 설정
                        file_path = os.path.join(directory_path, file_name)
                        
                        # CSV 파일 읽기
                        read_test = pd.read_csv(file_path)
                        
                        # 해당 그룹의 X 컬럼의 평균 값 읽기 및 저장
                        x_mean = np.mean(read_test['Delta_X']) # X, Delta_X(=절대값 아닌 것), Delta_X(=절대값 인것)
                        group_means[group].append(x_mean)

# 각 그룹별 평균 계산
for group in groups:
    group_means[group] = np.mean(group_means[group])

# 상관관계 계산을 위한 데이터프레임 생성
df_means = pd.DataFrame(group_means, index=[0]).T
df_means.columns = ['delta_X_Mean'] # D 그룹과 F그룹에서 NaN 값이 있음. 

# 성과 점수와 그룹 평균을 하나의 데이터프레임으로 결합
df_combined = pd.concat([df_means, performance_scores.rename('Performance_Score')], axis=1)

# 만약 NaN값이 있다면, 그 행은 건너 뛰고 상관관계 분석을 진행할 수 있도록 함. 
df_combined = df_combined.dropna()
#print(df_combined)

# 상관관계 및 p-value 계산
correlation, p_value = pearsonr(df_combined['delta_X_Mean'], df_combined['Performance_Score'])

print(f"Correlation: {correlation}") # -0.22011039
print(f"P-value:  {p_value}") # 0.7220264 || p-value 해석 => p_value 값이 작을수록 유의미한 결과가 있다고 볼 수 있음.


'''
X(=Pitch 축) 
Correlation : -0.22
P-value : 0.722

Delta_X(=Pitch 축)
Correlation : -0.823
P-value : 0.086

Delta_X(=Pitch 축) abs
Correlation :
P-value :

'''

# 산점도 시각화
plt.figure(figsize=(10, 6))

for group in df_combined.index:
    plt.scatter(df_combined.loc[group, 'delta_X_Mean'], df_combined.loc[group, 'Performance_Score'], label=f'Group {group}')

plt.xlabel('X Mean')
plt.ylabel('Performance Score')
plt.title(f'Scatter Plot between Group Performance Scores and delta_X Means\nCorrelation: {correlation:.2f}, P-value: {p_value:.2e}')
plt.legend()
plt.show()