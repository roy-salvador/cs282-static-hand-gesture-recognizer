import numpy as np
import cv2
import os
import skimage.feature as skf

GESTURE_CLASSES=[ 'A', 'B', 'D', 'E', 'F', 'K', 'L', 'N', 'W', 'Y']
DIR='training_set'
HOG_WINDOW_SIZE=88

# For HOG Feature
HOG_FEATURES_FILE='hog_training_set_' + str(HOG_WINDOW_SIZE) + '.csv'
hogFeaturesFile = open(HOG_FEATURES_FILE, "w")
  
# Process files in directory
for root, dirs, files in os.walk(DIR):
    for file in files:
        handgestureFile = os.path.join(root, file).replace('\\', '/')
        tag = file.split('_')[0]
        print 'Processing ' + handgestureFile
       
        # Extract features per frame
        cap = cv2.VideoCapture(handgestureFile)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                try :
                    graySkin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #smallerGray = cv2.resize(graySkin, (HOG_WINDOW_SIZE, HOG_WINDOW_SIZE))
 
                    # Find contours
                    ret,thresh = cv2.threshold(graySkin, 1,255,0)
                    _, contours,hierarchy = cv2.findContours(thresh, 1, 2)
                    
                   
                    if len(contours) != 0 :
                        cnt = contours[0]
                        for c in contours :
                            if len(cnt) < len(c) :
                                cnt = c
                        x,y,w,h = cv2.boundingRect(cnt)
                        
                        # Compute HOG
                        segmentedGraySkin = graySkin[y:y+h,x:x+w]
                        resizedSegmentedGraySkin = cv2.resize(segmentedGraySkin, (HOG_WINDOW_SIZE, HOG_WINDOW_SIZE)) 
                        hog = skf.hog(resizedSegmentedGraySkin, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False, normalise=True)
                        hogFeaturesFile.write(str(GESTURE_CLASSES.index(tag)) + ',' + str(hog.tolist()).lstrip('[').rstrip(']') + '\n')
                        
                    
                except:
                    None
            else :
                break
            
        cap.release()
cv2.destroyAllWindows()
hogFeaturesFile.close()