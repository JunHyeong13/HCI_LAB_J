import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 비디오 파일 경로
video_path = 'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2_central.mp4'

cap = cv2.VideoCapture(video_path)

# 비디오 프레임 수 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)

# 특정 구간 설정 (예: 329초 ~ 330초)
start_seconds = 329  # 시작할 시간
end_seconds = 330    # 끝낼 시간
total_seconds = end_seconds - start_seconds

# 프레임 단위로 총 프레임 수 계산
start_frame = int(start_seconds * fps)
end_frame = int(end_seconds * fps)
total_frames = end_frame - start_frame

# 시간 배열 생성
years = np.linspace(start_seconds, end_seconds, total_frames)

# 엑셀 데이터 불러오기
data_xlse = pd.read_excel('C:/Users/user/Downloads/Face Video/Face_1W_A1_S2_central.xlsx')
data = data_xlse['box.center_y']

# 특정 구간의 데이터 추출
data_trimmed = data[start_frame:end_frame]

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(years, data_trimmed, lw=2)

# y축 범위 설정
y_min = 0
y_max = np.ceil(data_trimmed.max())
ax.set_ylim(y_min, y_max)

# x축 눈금 설정
xticks = np.arange(start_seconds, end_seconds + 0.1, 0.1)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks.astype(int))
plt.xticks(rotation=45, ha='right')
ax.set_xlim(start_seconds, end_seconds)

# 그래프 제목 및 라벨 설정
ax.set_title('1 min Y axis graph (Zoomed)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('box.center_y')

# 그래프 보여주기
plt.show()

# 캡처 해제
cap.release()
