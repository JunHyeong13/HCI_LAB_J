#Pitch(=X axis) abs 값을 주차 별로 평균 내어 저장하는 코드
 
import os
import pandas as pd
import numpy as np

# 경로에 존재하는 파일 읽어오기. 
Weeks = ['1W', '2W', '3W', '4W']
Steps = ['S1', 'S2']
groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

path = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/'

#그룹 별 평균 값을 저장하기 위한 딕셔너리 선언 
group_means = {group: {week: [] for week in Weeks} for group in groups}

for group in groups:
    for week in Weeks:
        for i in ['1', '2', '3', '4']:
            for step in Steps:
                # Set the directory path
                directory_path = os.path.join(path, f'{group}_group_delta/')
                # face_Synchrony/Save_File_delta/Face_{week}_{group}_{section}_(delta)_Synchrony.csv
                
                # Get the list of files in the directory
                if os.path.exists(directory_path):
                    file_list = os.listdir(directory_path)
                    
                    for file_name in file_list:
                        # Check if the file name contains the specific week, group, and step
                        if f'Face_{week}_{group}{i}_{step}' in file_name:
                            # Set the file path
                            file_path = os.path.join(directory_path, file_name)
                            read_test = pd.read_csv(file_path)
                            
                            # Calculate the mean of the 'Delta_X' column
                            x_mean = np.mean(read_test['Delta_X'])
                            print(f"Current {week}_{group}{i}_{step} mean : ", x_mean)
                            
                            # Append the mean to the corresponding group and week
                            group_means[group][week].append(x_mean)
                            #print(group_means)

#NaN 값이 있을 시, 평균 값이 계산되지 않으므로, nan 부분이 있다면 넘어갈 수 있도록 세팅.
weekly_means = {group: {week: np.nanmean(values) if len(values) > 0 else np.nan for week, values in group_means[group].items()} for group in groups}

weekly_means_df = pd.DataFrame(weekly_means).T
output_path = 'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/group_weekly_delta_means.xlsx'
weekly_means_df.to_excel(output_path)
print(weekly_means_df)


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