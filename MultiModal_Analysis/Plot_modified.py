import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 비디오 파일 불러오는 경로 
videos = [
    'C:/Users/user/Downloads/Face Video/A1_result/Distance_A1.mp4',
    'C:/Users/user/Downloads/Face Video/A2_result/Distance_A2.mp4',
    'C:/Users/user/Downloads/Face Video/A3_result/Distance_A3.mp4',
    'C:/Users/user/Downloads/Face Video/A4_result/Distance_A4.mp4'
]

# CSV 파일 불러오는 경로 
csv_files = [
    'C:/Users/user/Downloads/Face Video/A1_result/distance_values_A1.csv',
    'C:/Users/user/Downloads/Face Video/A2_result/distance_values_A2.csv',
    'C:/Users/user/Downloads/Face Video/A3_result/distance_values_A3.csv',
    'C:/Users/user/Downloads/Face Video/A4_result/distance_values_A4.csv'
]


'''
# 끄덕임을 보는 코드 구간. 
videos = [
    'C:/Users/user/Downloads/Face Video/A1_result/Face_1W_A1_S2_HeadRotation_delta.mp4',
    'C:/Users/user/Downloads/Face Video/A2_result/Face_1W_A2_S2_HeadRotation_delta.mp4',
    'C:/Users/user/Downloads/Face Video/A3_result/Face_1W_A3_S2_HeadRotation_delta.mp4',
    'C:/Users/user/Downloads/Face Video/A4_result/Face_1W_A4_S2_HeadRotation_delta.mp4'
]

# CSV 파일 불러오는 경로 
csv_files = [
    'C:/Users/user/Downloads/Face Video/A1_result/head_pose_coordinates_delta_A1.csv',
    'C:/Users/user/Downloads/Face Video/A2_result/head_pose_coordinates_delta_A2.csv',
    'C:/Users/user/Downloads/Face Video/A3_result/head_pose_coordinates_delta_A3.csv',
    'C:/Users/user/Downloads/Face Video/A4_result/head_pose_coordinates_delta_A4.csv'
]
'''

# Open video capture objects
cap = [cv2.VideoCapture(video_path) for video_path in videos]


# Create individual windows for each video feed
windows = ['Video 1', 'Video 2', 'Video 3', 'Video 4']
for window_name in windows:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

'''
# 비디오 프레임 수 가져오기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
'''

# 비디오 프레임 수 가져오기
fps = cap[0].get(cv2.CAP_PROP_FPS)
start_seconds = 180 # 6분
end_seconds =  210# 3분 25초 
total_frames = int(fps * (start_seconds - end_seconds))

# 나중에 자동화
data_xlse = []
for n in range(0, 4):
    cap[n].set(cv2.CAP_PROP_POS_FRAMES, (start_seconds * 25))
    data_1 = pd.read_csv(csv_files[n])
    data_xlse.append(data_1)


#x ,y, z
data = []
data_trimmed = []
plt.ion()
fig, ax = plt.subplots(4, 1, figsize=(10, 15))

for n in range(0, 4):
    data.append(data_xlse[n]['Distance_cm'])
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    data_trimmed.append(data[n][start_frame:end_frame])
    print(data_trimmed[n])
    line, = ax[n].plot(np.arange(start_frame, end_frame), data_trimmed[n], lw=2)

for n in range(0, 4):
    y_min = np.floor(10.0)
    y_max = np.ceil(60.0)
    ax[n].set_ylim(y_min, y_max)

    # x 축 설정
    xticks = np.arange(start_seconds, end_seconds + 1, 1)  # 5초 간격으로 눈금 설정
    xtick_positions = np.arange(start_frame, end_frame + 1, int(fps))  # 6개의 위치 눈금 (end_frame - start_frame) // 6
    ax[n].set_xticks(xtick_positions)
    ax[n].set_xticklabels(xticks)
    plt.setp(ax[n].get_xticklabels(), rotation=45, ha='right')
    ax[n].set_xlim(start_frame, end_frame)

plt.draw()
plt.pause(0.001)

# 끄덕임이 있는 시간 때에 선을 그리기
# red_lines_246 = [ax[n].axvline(x=start_frame + 246 - start_seconds * fps, color='b', linestyle='--') for n in range(4)]
# red_lines_247 = [ax[n].axvline(x=start_frame + 247 - start_seconds * fps, color='b', linestyle='--') for n in range(4)]

red_lines = [ax[n].axvline(x=0, color='r') for n in range(4)]


plt.ion()
plt.show()
frame_num = start_seconds * fps
frame_end = end_seconds * fps

ret = [None for _ in range(4)]  # Initialize ret as a list of None values
frame = [None for _ in range(4)]  # Initialize frame as a list of None values

while frame_num < frame_end:

    all_ret = True
    all_frames = []

    for n in range(4):
        ret, frame = cap[n].read()
        all_ret = all_ret and ret
        all_frames.append(frame)

        if not all_ret:
            break

    # 현재 시간이 초 단위가 되도록 계산
    current_time_sec = frame_num

    for line in red_lines:
        line.set_xdata([current_time_sec])
        
    # 현재 시간 텍스트 추가
    #time_text = f'Time: {current_time_sec:.2f} sec'
    
    # 비디오 프레임을 OpenCV로 보여주기
    for n in range(0, 4):
        #cv2.putText(all_frames[n], time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(windows[n], all_frames[n])
    
    # 플롯 업데이트
    plt.draw()
    plt.pause(1/fps)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

for n in range(4):
    cap[n].release()

for window_name in windows:
    cv2.destroyWindow(window_name)

