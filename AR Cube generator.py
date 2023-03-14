import cv2
import numpy as np
import math
import glob
from matplotlib import pyplot as plt
import pupil_apriltags

 # loading mtx data from problem number 3 
with np.load('CameraParams.npz') as file:
    mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]
def bounding_box(mtx, x, y, r):

   #Estimating rotation and translation matrix suing functions in pupil apriltag

    pose = r.pose_R
    pose = np.concatenate
    T_mtx = r.pose_t

    intrinsic_mtx_1 = np.append((pose, T_mtx), axis = 1)
    
    p = mtx * intrinsic_mtx_1
    X = [(x), (y), (0), (1)]

    P = p * X

    return P

camera_matrix = mtx

detector = pupil_apriltags.Detector(families='tag36h11')
cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    gray_apriltag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_apriltag = detector.detect(gray_apriltag, estimate_tag_pose = True, camera_params = (1.05766726e+03, 1.05736759e+03, 6.27459926e+02, 4.74663319e+02), tag_size = 0.076)
    
    for r in result_apriltag:
        (A, B, C, D) = r.corners
        Corner1 = (int(A[0]), int(A[1]))
        Corner2 = (int(B[0]), int(B[1]))
        Corner3 = (int(C[0]), int(C[1]))
        Corner4 = (int(D[0]), int(D[1]))
        
        

        cv2.line(image, Corner1, Corner2, (255, 255, 255), 3)
        cv2.line(image, Corner2, Corner3, (255, 255, 255), 3)
        cv2.line(image, Corner3, Corner4, (255, 255, 255), 3)
        cv2.line(image, Corner4, Corner1, (255, 255, 255), 3)

        # Box above the April Tag

        # Lines from the April Tag
        cv2.line(image, Corner1, (Corner1[0], Corner1[1] - 75), (255, 255, 255), 3)
        cv2.line(image, Corner2, (Corner2[0], Corner2[1] - 75), (255, 255, 255), 3)
        cv2.line(image, Corner3, (Corner3[0], Corner3[1] - 75), (255, 255, 255), 3)
        cv2.line(image, Corner4, (Corner4[0], Corner4[1] - 75), (255, 255, 255), 3)

        # Top cover for the box
        cv2.line(image, (Corner1[0], Corner1[1] - 75), (Corner2[0], Corner2[1] - 75), (255, 255, 0), 3)
        cv2.line(image, (Corner2[0], Corner2[1] - 75), (Corner3[0], Corner3[1] - 75), (255, 255, 0), 3)
        cv2.line(image, (Corner3[0], Corner3[1] - 75), (Corner4[0], Corner4[1] - 75), (255, 255, 0), 3)
        cv2.line(image, (Corner4[0], Corner4[1] - 75), (Corner1[0], Corner1[1] - 75), (255, 255, 0), 3)

        tag_number = r.tag_family.decode("utf-8")
        cv2.putText(image, tag_number, (Corner1[0], Corner1[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if cv2.waitKey(1) == 113:
        break
    
    cv2.imshow("Live Video", image)
    
