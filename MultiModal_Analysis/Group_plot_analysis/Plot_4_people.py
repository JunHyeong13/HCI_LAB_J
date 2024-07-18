import cv2
import mediapipe as mp
import numpy as np
import csv

# MediaPipe 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 웹캡 캡처 객체 초기화
cap = cv2.VideoCapture('C:/Users/user/Downloads/Face Video/A_1W.mp4')
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Specify the codec to use
output_video = cv2.VideoWriter('C:/Users/user/Downloads/Face Video/Four_A/Distance_All.mp4', fourcc, 25.0, (int(width), int(height)))  # Filename, codec, FPS, frame size

# CSV 파일 초기화
csv_file = 'C:/Users/user/Downloads/Face Video/Four_A/Distance_values_all.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Face_ID', 'Distance_cm'])  # 헤더 작성

# NORMALIZED_FOCAL_X = 1.40625
NORMALIZED_FOCAL_X = 1.1
IRIS_WIDTH_CM = 1.17  # Adjusted to centimeters

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    height, width, _ = frame.shape

    # Draw face mesh landmarks
    if results.multi_face_landmarks:
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            print(face_id)
            iris_left_min_x = float('inf')
            iris_left_max_x = float('-inf')
            
            for point in mp_face_mesh.FACEMESH_LEFT_IRIS:
                x = face_landmarks.landmark[point[0]].x * width

                if x < iris_left_min_x:
                    iris_left_min_x = x
                if x > iris_left_max_x:
                    iris_left_max_x = x
                    
            dx = iris_left_max_x - iris_left_min_x
            fx = min(width, height) * NORMALIZED_FOCAL_X
            dZ = (fx * (IRIS_WIDTH_CM / dx)) if dx > 0 else 0.0

            # Draw face mesh landmarks for the face
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            # Draw iris landmarks
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )

            # Draw the calculated distance on the frame
            cv2.putText(
                frame,
                f"Face ID {face_id}: {dZ:.2f} cm",
                (10, 70 + 30 * face_id),  # Adjusted to avoid overlapping with the time text
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            
            # Save distance to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, face_id, dZ])

    # Calculate current time
    current_time_sec = frame_count / fps
    
    cv2.putText(
        frame,
        f"Time: {current_time_sec:.2f} sec",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    cv2.imshow("MediaPipe Face Mesh", frame)
    output_video.write(frame)
    
    frame_count += 1
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()
output_video.release()
cv2.destroyAllWindows()