import cv2
import numpy as np
import math

def kp_detection(image):
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(image, None)

    img  = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, descriptors, img

def match_keypoints(descriptors1, descriptors2, keypoints1, keypoints2, image, MIN_MATCH_DIST, MAX_ANGLE):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < MIN_MATCH_DIST * n.distance:
            good_matches.append(m)

    # Build new list instead of removing while iterating
    filtered_matches = []
    for match in good_matches:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]

        x1, y1 = kp1.pt
        x2, y2 = kp2.pt

        angle = math.atan2( abs(y2 - y1), abs(x2 - x1) ) * 180 / math.pi
        print(f"Match: ({x1:.2f}, {y1:.2f}) <-> ({x2:.2f}, {y2:.2f}), Angle: {angle:.2f} degrees")
        if (angle <= MAX_ANGLE) and (image[int(y1), int(x1)][0] < 240 or image[int(y1), int(x1)][1] < 240 or image[int(y1), int(x1)][2] < 240):
            filtered_matches.append(match)
            

    return filtered_matches
