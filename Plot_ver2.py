import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

# 비디오 파일 경로
video_path = 'C:/Users/나비/Downloads/Test_Video/Face_1W_A1_S3_landmark.mp4'
cap = cv2.VideoCapture(video_path)

'''
# 비디오 프레임 수 가져오기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
'''

# 비디오 프레임 수 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
total_seconds = 10  # 첫 10초만 재생
total_frames = int(fps * total_seconds)

years = np.arange(0, total_seconds, 1/fps)  # 1초 단위가 아닌 프레임 단위
data_xlse = pd.read_excel('C:/Users/나비/Downloads/Test_Excel/Face_1W_A1_S3.xlsx')
data = data_xlse['box.width']

# 데이터 10초 가정임. 전체를 돌릴때는 지우셈
data_trimmed = data[:len(years)]
fig, ax = plt.subplots()
line, = ax.plot(years, data_trimmed, lw=2)  

#이건 전체 데이터를 돌릴 때
'''
fig, ax = plt.subplots()
line, = ax.plot(years, data, lw=2)
'''

y_min = np.floor(data.min())
y_max = np.ceil(data.max())
ax.set_ylim(y_min, y_max)

# x 축을 5초 단위로 눈금 설정
xticks = np.arange(0, total_seconds + 1, 1)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
plt.xticks(rotation=45, ha='right')
ax.set_xlim(0, total_seconds)

# 빨간색 수직선의 초기화
red_line = ax.axvline(x=0, color='r')

# 플롯 초기화 함수
def init():
    red_line.set_xdata([0])
    return red_line,

plt.ion()
plt.show()
frame_num = 0

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