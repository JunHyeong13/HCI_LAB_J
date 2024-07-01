import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

# 비디오 파일 경로
video_path = 'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2_central.mp4' # C:\Users\user\Downloads\Face Video
cap = cv2.VideoCapture(video_path)

'''
# 비디오 프레임 수 가져오기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
'''

# 비디오 프레임 수 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
# total_seconds = 300   # 첫 10초만 재생 || 60초 * 20분 = 1200

start_seconds = 240 # 시작할 부분의 시간
end_seconds =  300# 끝낼 부분의 시간
total_seconds = end_seconds - start_seconds # 보여줄 전체 시간

total_frames = int(fps * total_seconds)
# 전체 시간을 기준으로 보여줌
#years = np.arange(0, total_seconds, 1/fps) 

# 특정 시간에 맞게 보여줌.
years = np.arange(0, total_seconds, 1/fps)

# 1초 단위가 아닌 프레임 단위
data_xlse = pd.read_excel('C:/Users/user/Downloads/Face Video/Face_1W_A1_S2_central.xlsx') # C:\Users\user\Downloads\Face Video
data = data_xlse['box.center_y']


# 특정 구간을 보여주기 위해 선언.
start_frame = int(start_seconds * fps)
end_frame = int(end_seconds * fps)
data_trimmed = data[start_frame:end_frame]

#data_trimmed = data[:len(years)] # 전체 시간에 한하여 보여줄 수 있도록 함.
fig, ax = plt.subplots(figsize=(12,5))
line, = ax.plot(years, data_trimmed, lw=2)  

#이건 전체 데이터를 돌릴 때
'''
fig, ax = plt.subplots()
line, = ax.plot(years, data, lw=2)
'''
'''
y_min = np.floor(data.min()) # 주어진 숫자의 소수점 이하를 버리고, 정수 부분만 남기는 함수
y_max = np.ceil(data.max()) # 인수로 받은 숫자를 반올림하여 반환.
'''
y_min = 0 # 주어진 숫자의 소수점 이하를 버리고, 정수 부분만 남기는 함수
y_max = np.ceil(data.max()) # 인수로 받은 숫자를 반올림하여 반환.

ax.set_ylim(y_min, y_max) # set_ylim의 경우, y축의 최솟값, 최댓값을 설정. 

# x 축을 5초 단위로 눈금 설정
xticks = np.arange(0, total_seconds + 1, 5)
ax.set_xticks(xticks)
#ax.set_xticklabels(xticks)
ax.set_xticklabels((xticks + start_seconds).astype(int))
plt.xticks(rotation=45, ha='right')
ax.set_xlim(0, total_seconds)

# 빨간색 수직선의 초기화
red_line = ax.axvline(x=0, color='r') 

# 플롯 초기화 함수
def init():
    red_line.set_xdata([0]) # 0
    return red_line,

plt.ion()
plt.show()
frame_num = 0

# 비디오 캡처 시작 위치 설정
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while cap.isOpened() and frame_num < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 시간이 초 단위가 되도록 계산
    current_time_sec = frame_num / fps
    # 빨간색 수직선의 x 좌표 업데이트
    red_line.set_xdata([current_time_sec])
    
    # 비디오 프레임을 OpenCV로 보여주기
    cv2.imshow('Video', frame)
    
    # 플롯 업데이트
    plt.draw()
    plt.pause(1/fps)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()