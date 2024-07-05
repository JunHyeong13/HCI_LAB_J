import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 비디오 파일 불러오는 경로 
videos = [
    'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2.mp4',
    'C:/Users/user/Downloads/Face Video/Face_1W_A2_S2.mp4',
    'C:/Users/user/Downloads/Face Video/A3_result/Face_1W_A3_S2.mp4',
    'C:/Users/user/Downloads/Face Video/A4_result/Face_1W_A4_S2.mp4'
]

# CSV 파일 불러오는 경로 
csv_files = [
    'C:/Users/user/Downloads/Face Video/A1_result/head_pose_coordinates_delta_A1.csv',
    'C:/Users/user/Downloads/Face Video/A2_result/head_pose_coordinates_delta_A2.csv',
    'C:/Users/user/Downloads/Face Video/A3_result/head_pose_coordinates_delta_A3.csv',
    'C:/Users/user/Downloads/Face Video/A4_result/head_pose_coordinates_delta_A4.csv'
]

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
start_seconds = 240
end_seconds = 270
total_frames = int(fps * (start_seconds - end_seconds))

years = np.arange(start_seconds, end_seconds, 1/fps)  # 1초 단위가 아닌 프레임 단위

# 나중에 자동화
data_xlse = []
for n in range(0, 4):
    data_1 = pd.read_csv(csv_files[n])
    print(data_1)
    data_xlse.append(data_1)

#x ,y, z
data = []
data_trimmed = []
plt.ion()
fig, ax = plt.subplots(4, 1)

for n in range(0, 4):
    data.append (data_xlse[n]['X'])
    data_trimmed.append (data[n][:len(years)])
    line, = ax[n].plot(years, data_trimmed[n], lw=2)  

#이건 전체 데이터를 돌릴 때
'''
fig, ax = plt.subplots()
line, = ax.plot(years, data, lw=2)
'''

for n in range(0, 4):
    y_min = np.floor(-5)
    y_max = np.ceil(15)
    ax[n].set_ylim(y_min, y_max)

    # x 축을 5초 단위로 눈금 설정
    xticks = np.arange(start_seconds, end_seconds + 1, 1)
    ax[n].set_xticks(xticks)
    ax[n].set_xticklabels(xticks)
    plt.xticks(rotation=45, ha='right')
    ax[n].set_xlim(start_seconds, end_seconds)


red_lines = [ax[n].axvline(x=0, color='r') for n in range(4)]

plt.ion()
plt.show()
frame_num = start_seconds * fps

ret = [None for _ in range(4)]  # Initialize ret as a list of None values
frame = [None for _ in range(4)]  # Initialize frame as a list of None values

while True:

    all_ret = True
    all_frames = []

    for n in range(4):
        ret, frame = cap[n].read()
        all_ret = all_ret and ret
        all_frames.append(frame)

        if not all_ret:
            break

    # 현재 시간이 초 단위가 되도록 계산
    current_time_sec = frame_num / fps

    for line in red_lines:
        line.set_xdata([current_time_sec])
    
    # 비디오 프레임을 OpenCV로 보여주기
    for n in range(0, 4):
        cv2.imshow(windows[n], all_frames[n])
    
    # 플롯 업데이트
    plt.draw()
    plt.pause(1/fps)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

for n in range(0, 4):
    print(n)
    cap[n].release()
    cv2.destroyAllWindows(windows[n])

cv2.destroyAllWindows()