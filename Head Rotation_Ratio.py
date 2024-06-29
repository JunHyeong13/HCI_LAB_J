import cv2
import mediapipe as mp
import numpy as np
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks = True)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# C:/Users/user/Downloads/Face Video/
video_path = 'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2.mp4'
cap = cv2.VideoCapture(video_path)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Specify the codec to use
output_video = cv2.VideoWriter('C:/Users/user/Downloads/Face Video/Result/Face_1W_A1_S2_HeadRotation.avi', fourcc, 25.0, (int(width), int(height)))  # Filename, codec, FPS, frame size


#왼쪽 눈
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
#오른쪽 눈
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]
R_H_LEFT = [33] 
R_H_RIGHT = [133]
L_H_LEFT = [362]
L_H_RIGHT = [263]

# 두 점 사이의 거리 계산(유클리드 거리 계산 부분)
def euclidean_distance(a, b):
    x1, y1, = a.ravel()
    x2, y2, = b.ravel()
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# 눈 ratio 값. iris = 눈동자 가운데, left(=눈의 좌측에 해당), Right(=눈의 우측에 해당)
def iris_positin (iris_center, right_point, left_point):
    center_to_right_distance = euclidean_distance(iris_center, right_point) # 눈동자와 눈의 오른쪽 부분까지의 거리. 
    total_distance = euclidean_distance(right_point, left_point) #눈 끝단의 왼쪽과 오른쪽 전체의 거리.
    ratio = center_to_right_distance / total_distance # 눈동자와 눈의 오른쪽 부분 / 눈 끝단의 왼쪽과 오른쪽 부분의 거리
    return ratio # 비율 값 반환. 

#cap = cv2.VideoCapture(0)

# 비디오 여는 부분. 
while cap.isOpened():
    success, image = cap.read()

    start = time.time() # 영상이 시작되는 시간.

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
        '''
        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections= mp_face_mesh.FACEMESH_CONTOURS, # 얼굴 주위에 메쉬를 그리는 부분.
            landmark_drawing_spec = drawing_spec,
            connection_drawing_spec = drawing_spec
        )
        '''
        
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

    center_left = np.array([l_cx, l_cy], dtype=np.int32)
    center_right = np.array([r_cx, r_cy], dtype=np.int32)
        
    cv2.circle(image, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(image, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

    ratio = iris_positin(
            center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0]
            )

    cv2.putText(image, f'Ratio: {ratio:.2f}', (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Head Pose Estimation', image)
    output_video.write(image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()