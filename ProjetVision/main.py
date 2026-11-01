import os

from pathlib import Path

from camera_calibration import calibration
from sift_kp_detection import kp_detection
from sift_kp_detection import match_keypoints
import numpy as np
import glob
import cv2
from resize_image import resize
from z_dimention import get_xyz_dimension
from threed_presentation import present_3d_points, presentation_2
from threed_presentation import show_boxes_from_xyz

project_dir = Path(__file__).resolve().parent
camera_matrix_path = project_dir / "camera_params" / "mtx.npy"

if camera_matrix_path.exists():
        print("Camera parameters already exist. Loading from file.")
        mtx = np.load(camera_matrix_path)
        print(mtx)
else:
        files = list((project_dir / "Images" / "4th test").glob("*.jpg"))

        print("Searching in:", project_dir / "Images" / "4th test")
        print("Number of images:", len(files))
        images = [cv2.imread(str(image)) for image in files]
        print("Number of images: ", len(images))
        mtx = calibration(images) 
        print(mtx)


img1_path = project_dir / "Images" / "im15.jpg"
img2_path = project_dir / "Images" / "im16.jpg"
img1 = cv2.imread(str(img1_path))
img2 = cv2.imread(str(img2_path))

if img1 is None:
        raise FileNotFoundError(f"Could not read image: {img1_path}")
if img2 is None:
        raise FileNotFoundError(f"Could not read image: {img2_path}")

keypoints1, descriptors1, img1_kp = kp_detection(img1)
keypoints2, descriptors2, img2_kp = kp_detection(img2)

img1_kp = resize(img1_kp)
img2_kp = resize(img2_kp)

cv2.imshow('img1', img1_kp)
cv2.waitKey(0)
cv2.imshow('img2', img2_kp)
cv2.waitKey(0)

MIN_MATCH_DIST = 0.8
MAX_ANGLE_DEGREE = 7
matches = match_keypoints(descriptors1, descriptors2, keypoints1, keypoints2,img1, MIN_MATCH_DIST, MAX_ANGLE_DEGREE)

img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches = resize(img_matches)
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)

presentation_2(keypoints1, keypoints2, matches, img1, img2)

b = 12
x,y,z = get_xyz_dimension(b, keypoints1, keypoints2, mtx, matches)


present_3d_points(x, y, z)

pythonbox_dimensions = np.array([
    [5.5, 5.5, 5.5],   # → assigned to cluster with lowest Z centroid
    [5.5, 10, 5.5],   # → assigned to cluster with middle Z centroid
    [10,  5.5, 5.5],   # → assigned to cluster with highest Z centroid
])

show_boxes_from_xyz(x, y, z, pythonbox_dimensions)