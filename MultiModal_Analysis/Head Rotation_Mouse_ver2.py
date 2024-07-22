import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pandas as pd
from tqdm import tqdm
import os


def saveFile(video_path, output_path, excel_path) :
    
    # 디렉터리 생성 로직 추가
    def create_directory_if_not_exists(file_path):
        parent = os.path.dirname(file_path)
        if not os.path.exists(parent):
            os.makedirs(parent)

    # 디렉터리 생성
    create_directory_if_not_exists(output_path)
    create_directory_if_not_exists(excel_path)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks = True)

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(video_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Specify the codec to use
    # output_video = cv2.VideoWriter(output_path, fourcc, 25.0, (int(width), int(height)))   #Filename, codec, FPS, frame size

    def euclidean_distance(a, b):
        x1, y1, = a.ravel()
        x2, y2, = b.ravel()
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    frame = 0
    data_x = []
    data_y = []
    data_z = []
    data_delta_x = []
    data_delta_y = []
    data_delta_z = []
    data_lip_distance = []
    data_time = []
    data_absence = False

    total_bar = tqdm(total=total_frames, position=0, leave=True)
    for frame in range(total_frames):
        total_bar.update(1)
        success, image = cap.read()

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
                    elif idx == 13:
                            upper_lip_point = np.array([lm.x, lm.y])
                    elif idx == 14:
                            lower_lip_point = np.array([lm.x, lm.y])

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
                
                lip_distance = euclidean_distance(upper_lip_point, lower_lip_point)

                # x,y,z, Delta_x, Delta_Y, Delta_ Z, lip_distance
                if frame == 0:
                    data_time.append(0.04 * (frame+1))
                    data_x.append(x)
                    data_y.append(y)
                    data_z.append(z)
                    data_delta_x.append(0)
                    data_delta_y.append(0)
                    data_delta_z.append(0)
                    data_lip_distance.append(lip_distance)
                else:
                    data_time.append(0.04 * (frame+1))
                    data_x.append(x)
                    data_y.append(y)
                    data_z.append(z)
                    
                    # 변화량 값을 저장하는 부분. => 이를 절대값으로 저장하여 보기 위해서는 abs()를 사용해주어야 함. 
                    data_delta_x.append(abs(x- data_x[frame-1]))
                    data_delta_y.append(abs(y- data_y[frame-1]))
                    data_delta_z.append(abs(z- data_z[frame-1]))
                    
                    # data_delta_x.append((x- data_x[frame-1]))
                    # data_delta_y.append((y- data_y[frame-1]))
                    # data_delta_z.append((z- data_z[frame-1]))
                    data_lip_distance.append(lip_distance)              
                '''
                mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                connections= mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
                )
                '''
            
            
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        else :
            # x,y,z, Delta_x, Delta_Y, Delta_ Z, lip_distance
            if frame == 0:
                data_time.append(0.04 * (frame+1))
                data_x.append(0)
                data_y.append(0)
                data_z.append(0)
                data_delta_x.append(0)
                data_delta_y.append(0)
                data_delta_z.append(0)
                data_lip_distance.append(0)
            else :
                data_time.append(0.04 * (frame+1))
                data_x.append(10000)
                data_y.append(10000)
                data_z.append(10000)
                data_delta_x.append(10000)
                data_delta_y.append(10000)
                data_delta_z.append(10000)
                data_lip_distance.append(10000)
        #cv2.imshow('Head Pose Estimation', image)
        #output_video.write(image)
        frame = frame + 1
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    #output_video.release()
    cv2.destroyAllWindows()
    csv_data = pd.DataFrame({
        'Time': data_time,
        'X': data_x,
        'Y': data_y,
        'Z': data_z,
        'Delta_X': data_delta_x,
        'Delta_Y': data_delta_y,
        'Delta_Z': data_delta_z,
        'Lip_Distance': data_lip_distance
    })
    csv_data = csv_data.replace(10000, np.nan)
    csv_data = csv_data.interpolate(method = 'values')
    # csv 파일로 저장하는 부분. 
    csv_data.to_csv(excel_path, index=False)


group_name = ['E','F','G'] #,'E','F','G'
weeks = ['1W','2W','3W','4W'] # , '2W', '3W', '4W'
participants_num = ['1','2','3','4'] # , '2', '3', '4'
section_num = ['S1','S2'] # , 'S2'


for group in group_name :
    for week in weeks:
        for participant in participants_num:
            for section in section_num: 
                video_path = f'D:/MultiModal/Data/Data_PreProcessing/Face/{group}_group/Face_{week}_{group}{participant}_{section}.mp4'
                #print(video_path)
                output_path = f'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/video/Face_{week}_{group}{participant}_{section}.mp4'
                #print(output_path)
                excel_path = f'D:/MultiModal/Data/Data_PreProcessing/Head_Rotation_Mouse/{group}_group_delta/Face_{week}_{group}{participant}_{section}.csv'
                #print(excel_path)
                saveFile(video_path, output_path, excel_path)

'''
video_path = 'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Videos/A/Face_1W_A2_S1.mp4'
output_path = 'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/Face_1W_A1_S2.mp4'
excel_path = 'D:/HCI_연구실_유재환/JaeHwanYou/AR Co/Synchrony/Education/Video/Plot Code/test/4명 영상 Test/data/A/Face_1W_A2_S1.csv'
if os.path.exists == False:
    saveFile(video_path, output_path, excel_path)
'''