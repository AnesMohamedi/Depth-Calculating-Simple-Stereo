import cv2
import numpy as np
import os
import glob
from resize_image import resize

def calibration(images):
    BOARD = (7,7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    objp = np.zeros((1, BOARD[0] * BOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2)
    nx = 7
    ny = 7

    print(objp)

    for image in images:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, (nx, ny), None, 
                                           cv2.CALIB_CB_ACCURACY)
    
        if ret == True:
            print("Chessboard corners found in image.")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            image = cv2.drawChessboardCorners(image, BOARD, corners2, ret)
            # Resize for display (1440p screen)
            img_resized = resize(image)
            cv2.imshow('img', img_resized)
            cv2.waitKey(0)
        else : 
            print("Chessboard corners not found in image.")
            continue
    
    _, mtx, _, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    os.makedirs("./camera_params", exist_ok=True)
    np.save("./camera_params/mtx", mtx)
    return mtx;

