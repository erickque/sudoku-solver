import numpy as np
import cv2

# Orders the coordinates of the corners left to right, top to bottom
def rectify(pts):
    pts = pts.reshape((4,2))
    new = np.zeros((4,2),dtype = np.float32)

    add = pts.sum(1)
    new[0] = pts[np.argmin(add)]
    new[2] = pts[np.argmax(add)]
     
    diff = np.diff(pts, axis=1)
    new[1] = pts[np.argmin(diff)]
    new[3] = pts[np.argmax(diff)]

    return new

def preprocess(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    thresh = cv2.adaptiveThreshold(blur,255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,11,2)
    img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(thresh,contours,1,(0,255,0),3)

    # Find the largest contour that is a square
    biggest = None
    max_area = 0
    # Image border is the largest contour. Not what is wanted
    border = max(contours, key = cv2.contourArea)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > max_area and len(approx)==4 and cnt is not border:
                biggest = approx
                max_area = area

    # Crop and perspective transform the image
    biggest=rectify(biggest)
    pts = np.float32([[0,0],[449,0],[449,449],[0,449]])
    M = cv2.getPerspectiveTransform(biggest,pts)
    dst = cv2.warpPerspective(blur,M,(450,450))
    return dst
