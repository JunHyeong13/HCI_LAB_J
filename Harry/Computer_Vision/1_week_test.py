import cv2 # import 진행
#print(cv2.__version__) # 버전 확인

import numpy as np

def main():
    img = cv2.imread('Dog.jpg') # 이미지 읽어오기
    cv2.imshow('img', img) # 이미지 출력
    cv2.waitKey(0) # 잠시 대기 
    
    if __name__ == '__main__': # main() 함수 호출
        main()