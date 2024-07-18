import cv2
import mediapipe as mp
import numpy as np
import time
import csv

mp_face_mesh = mp.solutions.face_mesh
# 랜드 마크 감지를 위한 함수
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 그리기 툴
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# C:/Users/user/Downloads/Face Video 'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/
video_path = 'C:/Users/user/Downloads/Face Video/A2_result/Face_1W_A2_S2.mp4'
cap = cv2.VideoCapture(video_path)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Specify the codec to use
output_video = cv2.VideoWriter('C:/Users/user/Downloads/Face Video/A2_result/Face_1W_A2_S2_HeadRotation_delta.mp4', fourcc, 25.0, (int(width), int(height)))  # Filename, codec, FPS, frame size

# CSV file로 값을 저장하기. 
csv_file_path = 'C:/Users/user/Downloads/Face Video/A2_result/head_pose_coordinates_delta_A2.csv'
frame_number = 0

# Initialize previous values for delta calculation (delta값 저장하는 부분.)
prev_x, prev_y, prev_z = None, None, None

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'X', 'Y', 'Z', 'Delta_X', 'Delta_Y', 'Delta_Z'])  # Write the header rotation


    while cap.isOpened():
        success, image = cap.read()

        start = time.time()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx ==263 or idx == 1 or idx == 61 or idx ==291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        
                        x, y = int(lm.x * img_w), int (lm.y * img_h)

                        # 2D 좌표
                        face_2d.append([x, y])

                        # 3D 좌표
                        face_3d.append([x, y, lm.z])
                    
                face_2d = np.array(face_2d, dtype = np.float64)
                face_3d = np.array(face_3d, dtype = np.float64)

                # 영상 matrix
                focal_length = 1 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h/2],
                                        [0, focal_length, img_w/2],
                                        [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype = np.float64)

                # PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # 회전 matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation angle
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # Display the direction (nose)
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # display text
                cv2.putText(image, 'Pitch : ' + str(np.round(x, 2)), (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 수평 축을 기준으로 얼굴을 위아래로 돌렸을 때. (가로 방향 위치)
                cv2.putText(image, 'Yaw : ' + str(np.round(y, 2)), (380, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 얼굴을 오른쪽으로 기울이면 양, 왼쪽으로 기울이면 음. (세로 방향 위치)
                cv2.putText(image, 'Roll : ' + str(np.round(z, 2)), (380, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 얼굴을 오른쪽으로 돌리면 양, 왼쪽으로 돌리면 음. (깊이)

                if prev_x is not None and prev_y is not None and prev_z is not None:
                    delta_x = x - prev_x
                    delta_y = y - prev_y
                    delta_z = z - prev_z
                else:
                    delta_x, delta_y, delta_z = 0, 0, 0
                
                prev_x, prev_y, prev_z = x, y, z
                #frame_number +=1   
                
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections= mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
            )
            
        writer.writerow([frame_number, np.round(x, 2), np.round(y, 2), np.round(z, 2), np.round(delta_x, 2), np.round(delta_y, 2), np.round(delta_z, 2)])
        frame_number +=1  
        cv2.imshow('Head Pose Estimation', image)
        output_video.write(image)
        

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
output_video.release()
cv2.destroyAllWindows()