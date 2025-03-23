# import cv2  # import 진행
# import numpy as np

# img_path = "/Users/jojunhyeong/Downloads/cv_sample/opencv-logo.png"
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지 읽어오기
    
# cv2.imshow('img', img) # 이미지 출력
# cv2.waitKey(0) # 잠시 대기 


import numpy as np  # 수치 계산과 배열 처리를 위한 라이브러리
import cv2         # 컴퓨터 비전 관련 기능을 제공하는 OpenCV 라이브러리

# 이미지 파일 경로 설정 (대상 이미지와 변환된 이미지)
img_path1 = '/Users/jojunhyeong/Downloads/cv_sample/target_img.png'
img_path2 = '/Users/jojunhyeong/Downloads/cv_sample/transformed_img.png'

# 파일 경로에서 이미지를 읽어옵니다 (컬러 이미지로 불러오기)
img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

# ORB(Oriented FAST and Rotated BRIEF) 객체 생성: 특징점 검출과 디스크립터(특징 기술자) 계산에 사용됩니다.
orb = cv2.ORB_create()

# 각 이미지에서 특징점과 디스크립터를 검출 및 계산합니다.
# kp1, kp2: 각 이미지의 특징점(keypoints)
# des1, des2: 각 특징점의 디스크립터(주변 특징을 표현하는 배열)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute-Force 매처 생성: 두 이미지의 디스크립터를 비교하여 가장 유사한 특징점들을 찾습니다.
# NORM_HAMMING은 ORB 디스크립터에 적합한 거리 측정 방법입니다.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 두 이미지의 디스크립터 간 매칭을 수행합니다.
matches = bf.match(des1, des2)

# 매칭 결과를 거리(distance) 기준으로 오름차순 정렬합니다.
# 거리가 짧을수록 더 유사한 특징점임을 의미합니다.
matches = sorted(matches, key=lambda x: x.distance)

# 상위 10개의 매칭 결과를 시각화합니다.
# 두 이미지에서 매칭된 특징점을 선으로 연결하여 결과 이미지(match_img)를 생성합니다.
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 매칭 결과를 확인하기 위해 창에 출력할 수 있습니다.
# cv2.imshow("Feature Matching", match_img)
# cv2.waitKey(0)

# 매칭된 특징점의 좌표를 추출합니다.
# queryIdx: img1에서의 특징점 인덱스, trainIdx: img2에서의 특징점 인덱스
pts1 = np.array([kp1[m.queryIdx].pt for m in matches[:10]], dtype=np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in matches[:10]], dtype=np.float32)

# RANSAC 알고리즘을 사용하여 두 이미지 간의 호모그래피(Homography) 행렬을 계산합니다.
# 호모그래피 행렬은 한 이미지의 평면을 다른 이미지의 평면으로 대응시키는 변환 행렬입니다.
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.5)
print(f"Homography matrix:\n{H}")

# 계산된 호모그래피 행렬을 이용하여 img1에 원근 변환을 적용합니다.
transformed_img = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

# 변환된 이미지와 원본 img2를 나란히 결합하여 결과 이미지를 만듭니다.
result = np.hstack((transformed_img, img2))

# 결과 이미지를 창에 출력합니다.
cv2.imshow("Result", result)
cv2.waitKey(0)