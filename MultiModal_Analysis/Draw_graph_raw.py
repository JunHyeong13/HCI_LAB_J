import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

# 그래프 하나 그려보기
video_path = 'C:/Users/HarryAnnie/Downloads/Video/Face_1W_A2_S2_landmark.mp4'
cap = cv2.VideoCapture(video_path)

# 비디오 프레임 수 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)

'''
# 전체 그래프를 그리고 싶을 때
total_seconds = 1200
'''
start_seconds = 5*60 # 시작할 부분의 시간
end_seconds = 6*60 # 끝낼 부분의 시간
total_seconds = end_seconds - start_seconds # 보여줄 전체 시간


total_frames = int(fps * total_seconds)
years = np.arange(0, total_seconds, 1/fps)

# 1초 단위가 아닌 프레임 단위
data_xlse = pd.read_excel('C:/Users/HarryAnnie/Downloads/File/Face_1W_A2_S2.xlsx') # head_pose_coordinates_A2.
data = data_xlse['box.center_y'] # box 중점의 y좌표 값을 기준.

#  보여줄 시간을 정했을 때, 그 부분에 대해서만 보여주기 위해 지정하는 부분.
#data_trimmed = data[:len(years)]

# 특정 구간을 보여주기 위해 선언.
start_frame = int(start_seconds * fps)
end_frame = int(end_seconds * fps)
data_trimmed = data[start_frame:end_frame] # start 시간부터 end 시간까지를 보여줌.

fig, ax = plt.subplots(figsize=(10,5))
line, = ax.plot(years, data_trimmed, lw=2)

#y_min = np.floor(data.min()) # 주어진 숫자의 소수점 이하를 버리고, 정수 부분만 남기는 함수
y_min = 0
#y_max = np.ceil(data.max()) # 인수로 받은 숫자를 반올림하여 반환.
y_max = 250 # 인수로 받은 숫자를 반올림하여 반환.

ax.set_ylim(y_min, y_max) # set_ylim의 경우, y축의 최솟값, 최댓값을 설정. 

# x 축을 30초 단위로 눈금 설정
xticks = np.arange(0, total_seconds + 1, 10)
ax.set_xticks(xticks)
#ax.set_xticklabels(xticks)
ax.set_xticklabels((xticks + start_seconds).astype(int)) # 시작하는 부분의 숫자를 x 레이블에 표시.
plt.xticks(rotation=45, ha='right')
ax.set_xlim(0, total_seconds)


# 그래프 제목 및 라벨 설정
ax.set_title('20 min box.center.y axis graph')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('box.center_y axis')

# ax.set_title('20 min X axis graph')
# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('X axis') 

# 그래프 보여주기
plt.show()

# 캡처 해제
cap.release()