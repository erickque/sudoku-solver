import numpy as np
import cv2

# Train the k-NN classifier on the training features and labels
def train():
    features_train = np.loadtxt('./train_data/trainfeatures.data',np.float32)
    labels_train = np.loadtxt('./train_data/trainlabels.data',np.float32)
    labels_train = labels_train.reshape((labels_train.size,1))

    clf = cv2.ml.KNearest_create()
    clf.train(features_train, cv2.ml.ROW_SAMPLE, labels_train)

    return clf

# Create the training features and labels
def process_data():
    img = cv2.imread('./train_data/train.png')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    img2, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    features =  np.empty((0,100))
    labels = []
    keys = [i for i in range(48,58)]

    # Number of fonts used in train.png
    fonts = 30
    count = 0

    for cnt in contours:
        # Check if the contour isn't a smaller contour within a number
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            # Check if the contour is a number
            if  h>28:
                # Take the region of interest (the number)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))

                # Assgins each feature its corresponding number
                labels.append(9 - int(count/fonts))
                feature = roismall.reshape((1,100))
                features = np.append(features,feature,0)
                count += 1

    labels = np.array(labels,np.float32)
    labels = labels.reshape((labels.size,1))
    print ("Training complete")

    np.savetxt('./train_data/trainfeatures.data',features)
    np.savetxt('./train_data/trainlabels.data',labels)

if __name__ == '__main__':
    process_data()
