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
group_performance_file = 'C:/Users/user/Desktop/Group_performance.xlsx'
# 주차 별, pitch x 값의 변화량을 average한 값. 
#total_synchrony_file = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/total_synchrony(delta)_1.xlsx'

# 얼굴 총 움직임량 값을 average한 값. 
#face_total = 'C:/Users/user/Documents/Face_Rotation_total.xlsx'

# 얼굴 몸 움직임량 값을 저장한 부분.
#face_movement = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_face_movement.xlsx'
#body_movement = 'C:/Users/user/Documents/Body_total_movement.xlsx'

lip_distance = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_lip_distance_means.xlsx'  

# plot을 저장하기 위한 부분.
save_path = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/'

# 데이터 불러오기
group_performance = pd.read_excel(group_performance_file)

#total_synchrony = pd.read_excel(total_synchrony_file)
# 얼굴 총 움직임량 값 불러오는 위치.
#total_face_rotation = pd.read_excel(face_total)
#total_face_movement = pd.read_excel(face_movement)

#total_body_movement = pd.read_excel(body_movement)
total_lip_distance = pd.read_excel(lip_distance)

# 그룹 이름을 인덱스로 설정
group_performance.set_index('Unnamed: 0', inplace=True)
#total_synchrony.set_index('Unnamed: 0', inplace=True)
#total_face_rotation.set_index('Unnamed: 0', inplace=True)
#total_face_movement.set_index('Unnamed: 0', inplace=True)
#total_body_movement.set_index('Unnamed: 0', inplace=True)
total_lip_distance.set_index('Unnamed: 0', inplace=True)

# 두 데이터프레임 병합(X 축에는 그룹 성과 점수)
#merged_data = group_performance.join(total_synchrony, lsuffix='_performance', rsuffix='_synchrony')
#merged_data = group_performance.join(total_face_rotation, lsuffix='_performance', rsuffix='_face_rotation')
#merged_data = group_performance.join(total_body_movement, lsuffix='_performance', rsuffix='_lip_distance')
merged_data = group_performance.join(total_lip_distance, lsuffix='_performance', rsuffix='_lip_distance')

'''
#y 축에는 총 얼굴 회전량 값을 넣어주기 위함.
'''
#merged_data = total_synchrony.join(total_face_rotation, lsuffix='_Synchrony', rsuffix='_face_movement')
#merged_data = total_face_movement.join(group_performance, lsuffix='_face_movement', rsuffix='_group_performance')

# NaN 값이 있는 행 제거
cleaned_data = merged_data.dropna()

# 각 주차별로 성능과 동기화 데이터 간의 상관관계 및 산점도 그리기
weeks = ['1W', '2W', '3W', '4W']

# x축과 y축 값의 범주 계산
'''
# x_min = cleaned_data[[f'{week}_performance' for week in weeks]].min().min()
# x_max = cleaned_data[[f'{week}_performance' for week in weeks]].max().max()
# y_min = cleaned_data[[f'{week}_face rotation' for week in weeks]].min().min()
# y_max = cleaned_data[[f'{week}_face rotation' for week in weeks]].max().max()

'''
cleaned_data.reset_index(inplace=True)

#그룹 이름에 None 값이 있는 경우 'Group'으로 대체
cleaned_data['Unnamed: 0'] = cleaned_data['Unnamed: 0'].fillna('Group')

# 데이터를 길게 변환
long_data = pd.DataFrame()

for week in weeks:
    temp_df = cleaned_data[[f'{week}_lip_distance', f'{week}_performance']].copy() # _performance
    temp_df.columns = ['lip_distance', 'Performance'] # 'Synchrony'
    temp_df['Week'] = week
    temp_df['Group'] = cleaned_data['Unnamed: 0']
    long_data = pd.concat([long_data, temp_df])

plt.figure(figsize=(14, 10))

# 산점도 그리기
scatter = sns.scatterplot(data=long_data, x='Performance', y='lip_distance', hue='Group', style='Week', palette='tab10', s=100)
sns.regplot(data=long_data, x='Performance', y='lip_distance', scatter=False)

# 각 주차별 상관관계 및 p-value 계산 및 주석 추가
for week in weeks:
    week_data = long_data[long_data['Week'] == week]
    corr, p_value = pearsonr(week_data['Performance'], week_data['lip_distance']) # face_rotation
    corr_text = f'{week}: r={corr:.2f}, p={p_value:.2e}'
    plt.annotate(corr_text, xy=(0.05, 0.95 - 0.05 * weeks.index(week)), xycoords='axes fraction', fontsize=12, 
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

'''
# 이전에 하나의 그래프가 아닌 4개의 그래프로 값을 따로 보여줄 때의 코드 
# # for i, week in enumerate(weeks, 1):
# #     x = cleaned_data[f'{week}_performance']
# #     y = cleaned_data[f'{week}_synchrony']
# #     #y = cleaned_data[f'{week}_face rotation']
    
# #     # 상관관계 및 p-value 계산
# #     corr, p_value = pearsonr(x, y)
# #     corr_text = f'{week}: r={corr:.2f}, p={p_value:.2e}'
    
# #     # 산점도 그리기
# #     plt.subplot(2, 2, i)
# #     #sns.scatterplot(x=x, y=y)
    
# #     # 각 그룹 명을 표시해주기 위해, 다음과 같이 설정. 
# #     # 하나의 그래프에 모두 표시 
# #     scatter = sns.scatterplot(x=x, y=y, hue=cleaned_data['Unnamed: 0'], style=week, palette='tab10', s=100)
# #     #sns.regplot(x=x, y=y, scatter=False, ax=plt.gca())
# #     sns.regplot(x=x, y=y, scatter=False, label=corr_text)
   
'''
   
'''
#plt 을 출력하기 위한 이름, 저장하기 위한 코드 정립. 
'''
plt.title('Scatter Plot of Group Performance vs. lip distance (1W to 4W)')
plt.xlabel('Performance')
plt.ylabel('lip distance') # face movement

'''
#x,y 축의 범위를 조정해줄 때 사용하는 코드.
#plt.xlim(10, 50)
#plt.ylim(0.0, 3.0) # (0.0, 3.0) (-0.005, 0.025)
'''

plt.legend(title='Group & Week', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Group_Performance_vs_lip_distance_all_weeks.png')) # body_movement_all
#plt.show()
plt.close()

#     plt.title(f'Scatter Plot of {week}_performance vs. {week}_synchrony')
#     #plt.title(f'Scatter Plot of {week}_performance vs. {week}_face rotation')
#     plt.xlabel(f'{week}_performance')
#     #plt.ylabel(f'{week}_synchrony')
#     plt.ylabel(f'{week}_face rotation')
#     #plt.ylim(y_min, y_max)
    
#     # 최대, 최솟 값 맞춰서 보여줄 수 있도록 함. 
#     # plt.xlim(x_min, x_max)
#     # plt.ylim(y_min, y_max)

#     plt.xlim(10, 50)
#     #plt.ylim(0.0, 3.0)
#     plt.ylim(-0.005, 0.025)
    
#     plt.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, 
#                  verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

#     # legend 제목 설정
#     handles, labels = scatter.get_legend_handles_labels()
#     scatter.legend(handles=handles, labels=labels, title='Group')

# plt.tight_layout()
# plt.savefig(os.path.join(save_path, 'Group_performance & group synchrony')) # group synchrony || face rotation
# plt.show()
# plt.close()
'''



# 주차별로 average()한 값을 출력해줄것.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 파일 경로
group_performance_file = 'C:/Users/user/Desktop/Group_performance.xlsx'
#total_synchrony_file = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/total_synchrony(delta)_1.xlsx'

# 얼굴 총 회전량 값을 average한 값. 
#face_total = 'C:/Users/user/Documents/Face_Rotation_total.xlsx'

# 얼굴 몸 움직임량 값을 저장한 부분.
#face_movement = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_face_movement.xlsx'

# 몸 움직임량 값을 저장한 부분. 
#body_movement = 'C:/Users/user/Documents/Body_total_movement.xlsx'

#얼굴 입 모양 값을 저장한 부분. 
lip_distance = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_lip_distance_means.xlsx'  

# plot을 저장하기 위한 부분.
save_path = 'D:/MultiModal/MultiModal_Model/Head_Rotation_Mouse/face_Synchrony/'

# 데이터 불러오기
group_performance = pd.read_excel(group_performance_file)
#total_synchrony = pd.read_excel(total_synchrony_file)
# 얼굴 총 회전량 값 불러오는 위치.
#total_face_rotation = pd.read_excel(face_total)
# 얼굴 몸 움직임량 값 불러오는 위치. 
#total_face_movement = pd.read_excel(face_movement)
#total_body_movement = pd.read_excel(body_movement)
total_lip_distance = pd.read_excel(lip_distance)

# 그룹 이름을 인덱스로 설정
group_performance.set_index(group_performance.columns[0], inplace=True)
#total_synchrony.set_index(total_synchrony.columns[0], inplace=True)
#total_face_rotation.set_index(total_face_rotation.columns[0], inplace=True)
#total_face_movement.set_index(total_face_movement.columns[0], inplace=True)
#total_body_movement.set_index(total_body_movement.columns[0], inplace=True)
total_lip_distance.set_index(total_lip_distance.columns[0], inplace=True)

# 두 데이터프레임 병합
#merged_data = group_performance.join(total_synchrony, lsuffix='_performance', rsuffix='_synchrony')
#merged_data = group_performance.join(total_face_movement, lsuffix='_performance', rsuffix='_face_movement')
#merged_data = total_synchrony.join(total_face_rotation, lsuffix='_Synchrony', rsuffix='_face_rotation')
#merged_data = total_synchrony.join(total_face_movement, lsuffix='_Synchrony', rsuffix='_face_movement')
#merged_data = group_performance.join(total_body_movement, lsuffix='_performance', rsuffix='_body_movement')
merged_data = group_performance.join(total_lip_distance, lsuffix='_performance', rsuffix='_lip_distance')

# NaN 값이 있는 행 제거
cleaned_data = merged_data.dropna()

# 각 주차별로 성능과 동기화 데이터 간의 상관관계 및 산점도 그리기
weeks = ['1W', '2W', '3W', '4W']

# 그룹별 주차 평균 계산
group_names = cleaned_data.index
avg_performance = cleaned_data[[f'{week}_performance' for week in weeks]].mean(axis=1)

#avg_synchrony = cleaned_data[[f'{week}_Synchrony' for week in weeks]].mean(axis=1)
#avg_face_rotation = cleaned_data[[f'{week}_face_rotation' for week in weeks]].mean(axis=1)
#avg_face_movement = cleaned_data[[f'{week}_face_movement' for week in weeks]].mean(axis=1)
#avg_body_movement = cleaned_data[[f'{week}_body_movement' for week in weeks]].mean(axis=1)
avg_lip_distance = cleaned_data[[f'{week}_lip_distance' for week in weeks]].mean(axis=1)

avg_data = pd.DataFrame({
    'Group': group_names,
    'Average Performance': avg_performance,
    #'Average Synchrony': avg_synchrony,
    #'Average face rotation' : avg_face_rotation,
    #'Average face movement' : avg_face_movement,
    #'Average body movement' : avg_body_movement.
    'Average lip distance' : avg_lip_distance
})

plt.figure(figsize=(10, 6))

# 산점도 그리기
#scatter = sns.scatterplot(data=avg_data, x='Average Performance', y='Average Synchrony', hue='Group', palette='tab10', s=100)
#scatter = sns.scatterplot(data=avg_data, x='Average Performance', y='Average face movement', hue='Group', palette='tab10', s=100)
#scatter = sns.scatterplot(data=avg_data, x='Average Synchrony', y='Average face movement', hue='Group', palette='tab10', s=100)
#scatter = sns.scatterplot(data=avg_data, x='Average Performance', y='Average body movement', hue='Group', palette='tab10', s=100)
scatter = sns.scatterplot(data=avg_data, x='Average Performance', y='Average lip distance', hue='Group', palette='tab10', s=100)

# 각 그룹별 평균값 주석 추가
for index, row in avg_data.iterrows():
    group = row['Group']
    performance_mean = row['Average Performance']
    #synchrony_mean = row['Average Synchrony']
    #face_rotation_mean = row['Average face rotation'] # rotation
    #face_movement_mean = row['Average face movement']
    #body_movement_mean = row['Average body movement']
    lip_distance_mean = row['Average lip distance']
    annotation_text = f'{group}: ({performance_mean:.2f}, {lip_distance_mean:.2f})' # performance_mean
    plt.annotate(annotation_text, xy=(performance_mean, lip_distance_mean), # performance_mean
                 xytext=(5, 5), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

plt.title('Average Synchrony vs. lip distance by Group') # Synchrony
plt.xlabel('Average Performance') # Performance
plt.ylabel('Average lip distance') # Average Synchrony

#x,y 축에 대한 범위를 조정해주기 위한 코드 
#plt.xlim(10, 50)
#plt.ylim(0.0, 3.0) # -0.005, 0.025

plt.legend(title='Group')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Average_Group_performance_vs_lip_distance_by_groups.png')) # _synchrony_by_group.png'
#plt.show()
plt.close()


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

'''