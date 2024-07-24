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
#                 directory_path = os.path.join(path, f'{group}_group/')
                
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
#                             #print(x_mean)
                            
#                             # Append the mean to the corresponding group and week
#                             group_means[group][week].append(x_mean)

# #NaN 값이 있을 시, 평균 값이 계산되지 않으므로, nan 부분이 있다면 넘어갈 수 있도록 세팅.
# weekly_means = {group: {week: np.nanmean(values) if len(values) > 0 else np.nan for week, values in group_means[group].items()} for group in groups}

# weekly_means_df = pd.DataFrame(weekly_means).T
# output_path = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_means.xlsx'
# weekly_means_df.to_excel(output_path)
# print(weekly_means_df)

'''

# 상관분석 및 산점도 그래프 시각화 하는 부분.
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

performance_data_path = 'C:/Users/user/Desktop/Group_performance.xlsx' 
deltaX_data_path = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_means.xlsx' 

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
plt.savefig(os.path.join(save_path, 'correlation_bar_plot.png')) 
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
plt.savefig(os.path.join(save_path,'correlation_scatter_plots_with_regression_lines.png'))  # Save the scatter plots as a PNG file
#plt.show()
plt.close()


'''

# 그룹 별 어디에 분포되어 있는지 annotate 한 부분. 
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