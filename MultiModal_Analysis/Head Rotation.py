import cv2
import mediapipe as mp
import numpy as np
import time
import csv

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# C:/Users/user/Downloads/Face Video 'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/
video_path = 'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2.mp4'
cap = cv2.VideoCapture(video_path)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Specify the codec to use
output_video = cv2.VideoWriter('C:/Users/user/Downloads/Face VideoFace_1W_A1_S2_HeadRotation.avi', fourcc, 25.0, (int(width), int(height)))  # Filename, codec, FPS, frame size

# CSV file로 값을 저장하기. 
csv_file_path = 'head_pose_coordinates.csv'
frame_number = 0

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'X', 'Y', 'Z'])  # Write the header rotation


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
                cv2.putText(image, 'x: ' + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, 'y: ' + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, 'z: ' + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                writer.writerow([frame_number, np.round(x, 2), np.round(y, 2), np.round(z, 2)])
                    
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections= mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
            )

        cv2.imshow('Head Pose Estimation', image)
        output_video.write(image)
        
        frame_number +=1

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
output_video.release()
cv2.destroyAllWindows()