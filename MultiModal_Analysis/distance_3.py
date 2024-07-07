import cv2
import mediapipe as mp
import numpy as np
import csv

# MediaPipe 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 웹캠 캡처 객체 초기화
cap = cv2.VideoCapture('C:/Users/user/Downloads/Face Video/A4_result/Face_1W_A4_S2.mp4')
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks = True)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Specify the codec to use
output_video = cv2.VideoWriter('C:/Users/user/Downloads/Face Video/A4_result/Distance_A4.mp4', fourcc, 25.0, (int(width), int(height)))  # Filename, codec, FPS, frame size

# CSV 파일 초기화
csv_file = 'C:/Users/user/Downloads/Face Video/A4_result/distance_values_A4.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Distance_cm'])  # 헤더 작성


#NORMALIZED_FOCAL_X = 1.40625
NORMALIZED_FOCAL_X = 1.1
IRIS_WIDTH_CM = 11.7

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    height, width, _ = frame.shape
    iris_left_min_x = float('inf')
    iris_left_max_x = float('-inf')

    # Draw face mesh landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for point in mp_face_mesh.FACEMESH_LEFT_IRIS:
                x = face_landmarks.landmark[point[0]].x * width

                if x < iris_left_min_x:
                    iris_left_min_x = x
                if x > iris_left_max_x:
                    iris_left_max_x = x

            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_LEFT_IRIS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(48, 255, 48), thickness=1, circle_radius=1),
            )

    dx = iris_left_max_x - iris_left_min_x
    fx = min(width, height) * NORMALIZED_FOCAL_X
    dZ = (fx * (IRIS_WIDTH_CM / dx)) / 10.0 if dx > 0 else 0.0
    
    # Calculate current time
    current_time_sec = frame_count / fps

    # Draw the calculated distance on the frame
    cv2.putText(
        frame,
        f"distance : {dZ:.2f} cm",
        (int(width * 0.1), height-10),
        (0, 0, 255),
        2,
    )
    
    cv2.putText(
        frame,
        f"Time: {current_time_sec:.2f} sec",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    
    # Save distance to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_count, dZ])

    cv2.imshow("MediaPipe Face Mesh", frame)
    output_video.write(frame)
output_video.release()
cv2.destroyAllWindows()