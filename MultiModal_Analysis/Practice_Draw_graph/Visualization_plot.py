#Pitch(=X axis) abs 값을 주차 별로 평균 내어 저장하는 코드
''' 
# import os
# import pandas as pd
# import numpy as np

# # 경로에 존재하는 파일 읽어오기. 
# Weeks = ['1W', '2W', '3W', '4W']
# Steps = ['S1', 'S2']
# groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# path = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/'

# #그룹 별 평균 값을 저장하기 위한 딕셔너리 선언 
# group_means = {group: {week: [] for week in Weeks} for group in groups}

# for group in groups:
#     for week in Weeks:
#         for i in ['1', '2', '3', '4']:
#             for step in Steps:
#                 # Set the directory path
#                 directory_path = os.path.join(path, f'{group}_group_delta/')
                
#                 # Get the list of files in the directory
#                 if os.path.exists(directory_path):
#                     file_list = os.listdir(directory_path)
                    
#                     for file_name in file_list:
#                         # Check if the file name contains the specific week, group, and step
#                         if f'Face_{week}_{group}{i}_{step}' in file_name:
#                             # Set the file path
#                             file_path = os.path.join(directory_path, file_name)
#                             read_test = pd.read_csv(file_path)
                            
#                             # Calculate the mean of the 'Delta_X' column
#                             x_mean = np.mean(read_test['Delta_X'])
#                             print(f"Current {week}_{group}{i}_{step} mean : ", x_mean)
                            
#                             # Append the mean to the corresponding group and week
#                             group_means[group][week].append(x_mean)
#                             #print(group_means)

# #NaN 값이 있을 시, 평균 값이 계산되지 않으므로, nan 부분이 있다면 넘어갈 수 있도록 세팅.
# weekly_means = {group: {week: np.nanmean(values) if len(values) > 0 else np.nan for week, values in group_means[group].items()} for group in groups}

# weekly_means_df = pd.DataFrame(weekly_means).T
# output_path = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_delta_means.xlsx'
# weekly_means_df.to_excel(output_path)
# print(weekly_means_df)

'''

# 상관분석 및 산점도 그래프 시각화 하는 부분. (회귀선 포함하여 그래프 그리는 코드)
'''    
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

performance_data_path = 'C:/Users/user/Desktop/Group_performance.xlsx' 
deltaX_data_path = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_delta_means.xlsx' 

# save the plots
save_path = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/Graph_delta_abs/'

performance_data = pd.read_excel(performance_data_path)
deltaX_data = pd.read_excel(deltaX_data_path)

# Calculate the 'TOTAL' for deltaX data by summing across the weeks
deltaX_data['TOTAL'] = deltaX_data.iloc[:, 1:5].sum(axis=1)

# Rename columns for consistency
performance_data.columns = ['Group', '1W', '2W', '3W', '4W', 'TOTAL']
deltaX_data.columns = ['Group', '1W', '2W', '3W', '4W', 'TOTAL']

# Merge the two datasets on 'Group'
merged_data = pd.merge(performance_data, deltaX_data, on='Group', suffixes=('_score', '_deltaX'))

# Calculate Pearson correlation coefficients
correlation_results = {}
columns = ['1W', '2W', '3W', '4W', 'TOTAL']
for col in columns:
    correlation = merged_data[f'{col}_score'].corr(merged_data[f'{col}_deltaX'])
    correlation_results[col] = correlation

# Convert the results to a DataFrame for better display
correlation_df = pd.DataFrame(list(correlation_results.items()), columns=['Week', 'Correlation'])

# Display the correlation results
print(correlation_df)

# Define a color map for the groups
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
group_colors = {group: color for group, color in zip(merged_data['Group'].unique(), colors)}

# Create a bar plot for the correlation results
plt.figure(figsize=(10, 6))
plt.bar(correlation_df['Week'], correlation_df['Correlation'], color='skyblue')
plt.xlabel('Week')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('Correlation between Performance Scores and DeltaX by Week')
plt.ylim(-1, 1)
plt.axhline(0, color='gray', linestyle='--')
plt.savefig(os.path.join(save_path, 'correlation_bar_plot(detla).png')) 
#plt.show()
plt.close()

# 산점도 그래프 그려보는 구간. (각 주차별 및 전체)
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
weeks = ['1W', '2W', '3W', '4W', 'TOTAL']
axes = axes.flatten()

for i, week in enumerate(weeks):
    ax = axes[i]
    for group in merged_data['Group'].unique():
        group_data = merged_data[merged_data['Group'] == group]
        ax.scatter(group_data[f'{week}_deltaX'], group_data[f'{week}_score'], 
                   color=group_colors[group], label=group, alpha=0.6)
        
        
    # 회귀선 추가하는 부분. 
    x = merged_data[f'{week}_deltaX']
    y = merged_data[f'{week}_score']
    m, b = np.polyfit(x, y, 1)  # Fit a line to the data
    ax.plot(x, m*x + b, color='red', linestyle='--')
        
    ax.set_xlabel('DeltaX')
    ax.set_ylabel('Performance Score')
    ax.set_title(f'{week} Correlation: {correlation_results[week]:.2f}')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.savefig(os.path.join(save_path,'correlation_scatter_plots_with_regression_lines(delta).png'))  # Save the scatter plots as a PNG file
#plt.show()
plt.close()

'''  

'''
#face synchrony week값과 그룹의 성과 total 값간의 상관관계를 그리는 코드. 
# => 현재 코드에서는 전체 그룹의 주차를 평균 내어 그려진 것. 

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr

# # 파일 경로
# # file_csv = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/total_synchrony(delta)_1.xlsx'
# # file_excel = 'C:/Users/user/Desktop/Group_performance.xlsx'
# output = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/'

# # 파일 경로
# group_performance_file = 'C:/Users/user/Desktop/Group_performance.xlsx'
# total_synchrony_file = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/total_synchrony(delta)_1.xlsx'

# # 데이터 불러오기
# group_performance = pd.read_excel(group_performance_file)
# total_synchrony = pd.read_excel(total_synchrony_file)

# # 그룹 이름을 인덱스로 설정
# group_performance.set_index('Unnamed: 0', inplace=True)
# total_synchrony.set_index('Unnamed: 0', inplace=True)

# # 두 데이터프레임 병합
# merged_data = group_performance.join(total_synchrony, lsuffix='_performance', rsuffix='_synchrony')
# # 데이터 확인을 위해 csv 파일로 내보내기.
# #merged_data.to_csv(os.path.join(output, 'Merged_data.csv'))

# # NaN 값이 있는 행 제거
# cleaned_data = merged_data.dropna()
# #print(cleaned_data)

# # 여러 주차의 동기화 데이터를 한 번에 시각화하기 위해 melt 사용
# melted_data = cleaned_data.melt(id_vars=['TOTAL'], value_vars=['1W_synchrony', '2W_synchrony', '3W_synchrony', '4W_synchrony'], 
#                                 var_name='Week', value_name='Synchrony')


# # 성능(TOTAL)과 각 주차의 동기화 데이터 간의 상관관계 및 p-value 계산
# correlation_results = []
# for week in ['1W_synchrony', '2W_synchrony', '3W_synchrony', '4W_synchrony']:
#     corr, p_value = pearsonr(cleaned_data['TOTAL'], cleaned_data[week])
#     correlation_results.append((week, corr, p_value))

# # 상관관계 및 p-value 텍스트 생성
# correlation_texts = [f"{week}: r={corr:.2f}, p={p_value:.2e}" for week, corr, p_value in correlation_results]
# #print(correlation_texts)
# correlation_text = "\n".join(correlation_texts)

# # 산점도 그리기 (그룹 이름을 표시)
# plt.figure(figsize=(10, 8))
# sns.scatterplot(data=melted_data, x='TOTAL', y='Synchrony', hue='Week', style='Week', s=100)

# # 각 데이터 포인트에 그룹 이름 추가
# for i in range(cleaned_data.shape[0]):
#     plt.text(cleaned_data['TOTAL'].iloc[i], cleaned_data['1W_synchrony'].iloc[i], cleaned_data.index[i], horizontalalignment='right')
#     plt.text(cleaned_data['TOTAL'].iloc[i], cleaned_data['2W_synchrony'].iloc[i], cleaned_data.index[i], horizontalalignment='right')
#     plt.text(cleaned_data['TOTAL'].iloc[i], cleaned_data['3W_synchrony'].iloc[i], cleaned_data.index[i], horizontalalignment='right')
#     plt.text(cleaned_data['TOTAL'].iloc[i], cleaned_data['4W_synchrony'].iloc[i], cleaned_data.index[i], horizontalalignment='right')

# # 상관관계 및 p-value 텍스트 추가
# plt.annotate(correlation_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, 
#              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

# plt.title('Scatter Plot of Total Performance vs. Synchrony (1W to 4W)')
# plt.xlabel('Total Performance')
# plt.ylabel('Synchrony')
# plt.grid(True)
# plt.legend(title='Week')
# plt.show()

'''

'''
# face synchrony week값과 그룹의 성과 week 값 간의 상관관계를 그리는 코드. 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 파일 경로
file_csv = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/total_synchrony(delta).csv'
file_excel = 'C:/Users/user/Desktop/Group_performance.xlsx'
output = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/'

# CSV 파일 불러오기
df_csv = pd.read_csv(file_csv)

# Excel 파일 불러오기
df_excel = pd.read_excel(file_excel)

# 그룹 목록 추출
groups = df_excel['Unnamed: 0']

# 각 그룹별로 주차별 동기화 데이터 통합 (synchrony)읽어오는 곳 
group_week_data = {}
for group in groups:
    group_week_data[group] = {
        '1W': df_csv.filter(like=f'{group}_1W').mean(axis=1),
        '2W': df_csv.filter(like=f'{group}_2W').mean(axis=1),
        '3W': df_csv.filter(like=f'{group}_3W').mean(axis=1),
        '4W': df_csv.filter(like=f'{group}_4W').mean(axis=1)
    }

# 상관관계와 p-value 계산 및 산점도 그리기
results = []
fig, axes = plt.subplots(len(groups), 4, figsize=(20, 5 * len(groups)))

for i, group in enumerate(groups):
    df_group_weeks = pd.DataFrame(group_week_data[group])
    print(df_group_weeks)
    df_group_perf = df_excel[df_excel['Unnamed: 0'] == group].iloc[:, 1:].T
    df_group_perf.columns = [group]
    print(df_group_perf)
    
    # 공통 인덱스 찾기
    common_index = df_group_weeks.index.intersection(df_group_perf.index)
    df_group_weeks = df_group_weeks.loc[common_index]
    df_group_perf = df_group_perf.loc[common_index]
    
    for j, week in enumerate(['1W', '2W', '3W', '4W']):
        # 성과 데이터 추출
        performance_data = df_group_perf[group]
        
        if len(df_group_weeks[week]) > 1 and len(performance_data) > 1:
            # 동기화 데이터와 성과 데이터 간 상관관계와 p-value 계산
            corr, p_value = pearsonr(df_group_weeks[week], performance_data)
            results.append((group, week, corr, p_value))

            # 산점도 및 회귀선 그리기
            sns.scatterplot(ax=axes[i, j], x=df_group_weeks[week], y=performance_data, color='blue')
            sns.regplot(ax=axes[i, j], x=df_group_weeks[week], y=performance_data, scatter=False, color='red')
            axes[i, j].set_title(f'{group} {week} (r={corr:.2f}, p={p_value:.2g})')
        else:
            axes[i, j].set_title(f'{group} {week} (Insufficient data)')
        
        axes[i, j].set_xlabel('Synchronization')
        axes[i, j].set_ylabel('Performance')
        axes[i, j].set_xlim(-1, 1)  # x 축 범위를 -1에서 1로 설정

plt.tight_layout()
plt.show()

# 결과 데이터프레임 생성 및 출력
results_df = pd.DataFrame(results, columns=['Group', 'Week', 'Correlation', 'P-value'])
results_df.to_csv(os.path.join(output, 'Group_synchrony(Correlation).csv'))
print(results_df)

#'''

'''

# 그룹 별 어디에 분포되어 있는지 annotate 한 부분. 
'''
# for i, week in enumerate(weeks):
#     ax = axes[i]
#     ax.scatter(merged_data[f'{week}_deltaX'], merged_data[f'{week}_score'], color='blue', alpha=0.6)
#     ax.set_xlabel('DeltaX')
#     ax.set_ylabel('Performance Score')
#     ax.set_title(f'{week} Correlation: {correlation_results[week]:.2f}')
#     ax.axhline(0, color='gray', linestyle='--')
#     ax.axvline(0, color='gray', linestyle='--')
    
#     # Annotate each point with the group label
#     for idx, row in merged_data.iterrows():
#         ax.annotate(row['Group'], (row[f'{week}_deltaX'], row[f'{week}_score']), fontsize=15, alpha=0.7)

# # Adjust layout
# plt.tight_layout()
# plt.savefig(os.path.join(save_path,'correlation_scatter_plots.png'))  # Save the scatter plots as a PNG file
# #plt.show()
# plt.close()