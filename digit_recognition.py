import cv2
import numpy as np

# Train the k-NN classifier on the train features and labels
def train():
    features_train = np.loadtxt('./train_data/trainfeatures.data',np.float32)
    labels_train = np.loadtxt('./train_data/trainlabels.data',np.float32)
    labels_train = labels_train.reshape((labels_train.size,1))

    clf = cv2.ml.KNearest_create()
    clf.train(features_train, cv2.ml.ROW_SAMPLE, labels_train)

    return clf

def extract_grid(img):
    thresh = cv2.adaptiveThreshold(img,255,1,1,11,2)
    img2, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    clf = train()
    
    # A zero represents an empty box in the grid
    grid = np.zeros(shape=(9,9), dtype=int)
    
    for cnt in contours:
        # Check if the contour is the approximate size of a number
        if cv2.contourArea(cnt)>50 and cv2.contourArea(cnt)<1000:
            x,y,w,h = cv2.boundingRect(cnt)
            
            # Check dimension if the contour is a number
            if  h>15 and h<45 and w>5:    

                # Take the region of interest (the number)
                # And use the classifier to determine what it is 
                roi = thresh[y:y+h,x:x+w]
                roi = cv2.resize(roi,(10,10))
                roi = roi.reshape((1,100))
                roi = np.float32(roi)
                retval, results, neigh_resp, dists = clf.findNearest(roi,k=1)
                num = int(results[0][0])
                
                # Determine the row and col the number belongs to
                row = int(y/50)
                col = int(x/50)
                grid[row, col] = num
                
    grid = grid.tolist()            
    return grid
