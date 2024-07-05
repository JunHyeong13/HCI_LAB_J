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

video_path = 'A_1W.mp4'
cap = cv2.VideoCapture(video_path)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Specify the codec to use
output_video = cv2.VideoWriter('temp.mp4', fourcc, 25.0, (int(width/2), int(height/2)))  # Filename, codec, FPS, frame size

# CSV file로 값을 저장하기. 
csv_file_path = 'A1_1W.csv'
frame_number = 0

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame','PX','PY','PZ', 'RX', 'RY', 'RZ'])  # Write the header rotation


    while cap.isOpened():
        success, image = cap.read()

        start = time.time()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        half_width = int(width/2); half_height = int(height/2)
        image = image[half_height:,:half_width].copy()
      
        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        px,py,pz,rx,ry,rz = 0,0,0,0,0,0
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
                        if idx == 1:
                            px,py,pz = lm.x, lm.y, lm.z
                    
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
                rx = angles[0] * 360
                ry = angles[1] * 360
                rz = angles[2] * 360

                # Display the direction (nose)
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # display text
                cv2.putText(image, 'rx: ' + str(np.round(rx, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, 'ry: ' + str(np.round(ry, 2)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, 'rz: ' + str(np.round(rz, 2)), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(image, 'px: ' + str(np.round(px, 2)), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, 'py: ' + str(np.round(py, 2)), (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(image, 'pz: ' + str(np.round(pz, 2)), (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                
                
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections= mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
            )
        writer.writerow([frame_number, np.round(px, 4), np.round(py, 4), np.round(pz, 4), np.round(rx, 4), np.round(ry, 4), np.round(rz, 4)])
        frame_number +=1   
        cv2.imshow('Head Pose Estimation', image)
        if(frame_number %100 == 0):
            print(frame_number)
        output_video.write(image)
        
 
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
output_video.release()
cv2.destroyAllWindows()