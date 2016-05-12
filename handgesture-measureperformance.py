#
#	Hand Gesture Recognizer
#	@author Roy Salvador
#
import numpy as np
import argparse
import cv2
import skimage.feature as skf
import skimage.measure as skm
from sklearn.svm import LinearSVC
import sklearn.svm as svm
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
import os
from datetime import datetime
from sklearn.externals import joblib

GESTURE_CLASSES=['A', 'B', 'D', 'E', 'F', 'K', 'L', 'N', 'W', 'Y']
HOG_WINDOW_SIZE=88

# Train MultiClass SVM
TRAINING_FEATURES = []
TRAINING_LABELS = []
trainFile = open('model/hog_training_set_' + str(HOG_WINDOW_SIZE) + '.csv', 'r')
for line in trainFile:
    j=0
    featuresVector = []
    for value in line.strip().split(',') :
      if j==0 :
        TRAINING_LABELS.append(int(value))
      else :
        featuresVector.append(float(value)) 
      j = j+1
    TRAINING_FEATURES.append(featuresVector)	
trainFile.close()


print 'Initializing One vs Rest Multi Class SVM from saved training data'
print len(TRAINING_FEATURES[0])
print datetime.now().strftime('%Y-%m-%d %H:%M:%S')
OVR_SVM_CLF = OneVsRestClassifier(LinearSVC(random_state=0)).fit(TRAINING_FEATURES, TRAINING_LABELS)
print datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print 'Saving SVM model to hog_svm' + str(HOG_WINDOW_SIZE) + '.pkl'
joblib.dump(OVR_SVM_CLF, 'hog_svm_' + str(HOG_WINDOW_SIZE) + '.pkl') 


# Measure performance of dataset contained in some directoryL
def measurePerformance(datasetDIR) :
    GESTURE_COUNT=np.zeros([len(GESTURE_CLASSES), len(GESTURE_CLASSES)])
    print '================================================================================================'
    print datasetDIR
    # Process files in directory
    for root, dirs, files in os.walk(datasetDIR):
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
                        
                        # Find contours
                        ret,thresh = cv2.threshold(graySkin, 1,255,0)
                        _, contours,hierarchy = cv2.findContours(thresh, 1, 2)
                        
                       
                        if len(contours) != 0 :
                            cnt = contours[0]
                            for c in contours :
                                if len(cnt) < len(c) :
                                    cnt = c
                            x,y,w,h = cv2.boundingRect(cnt)
                            segmentedGraySkin = graySkin[y:y+h,x:x+w]
                            
                            # Compute HOG
                            resizedSegmentedGraySkin = cv2.resize(segmentedGraySkin, (HOG_WINDOW_SIZE, HOG_WINDOW_SIZE)) 
                            hog = skf.hog(resizedSegmentedGraySkin, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False, normalise=True)
                            temp = np.array(hog).reshape((1,len(hog)))
                            prediction = OVR_SVM_CLF.predict(temp)[0]
                            
                            #binarySkin = cv2.inRange(graySkin, 1, 255)
                            #hu = cv2.HuMoments(cv2.moments(binarySkin)).flatten()
                            #logTransformedHu = -np.sign(hu) * np.log10(np.abs(hu))
                            #temp = np.array(hog.tolist() + hu.tolist()).reshape((1,len(hog.tolist() + hu.tolist())))  
                            #prediction = OVR_SVM_CLF.predict(temp)[0]
                            
                            GESTURE_COUNT[prediction][GESTURE_CLASSES.index(tag)] = GESTURE_COUNT[prediction][GESTURE_CLASSES.index(tag)] + 1
                        
                    except:
                        None
                else :
                    break
            cap.release()
            
    # Print results; exclude background class
    print GESTURE_COUNT
    TOTAL_PER_CLASS = sum(GESTURE_COUNT)
    TOTAL_CORRECT=0
    i=0
    while i<len(GESTURE_CLASSES) :
        print GESTURE_CLASSES[i] + ' = ' + str(GESTURE_COUNT[i][i]) + '/' +  str(TOTAL_PER_CLASS[i]) + '(' + str(round(GESTURE_COUNT[i][i]*100.0/TOTAL_PER_CLASS[i],2)) + '%)' 
        TOTAL_CORRECT = TOTAL_CORRECT + GESTURE_COUNT[i][i]
        i=i+1
    print ''
    print 'OVERALL = ' + str(TOTAL_CORRECT) + '/' + str(sum(TOTAL_PER_CLASS)) + '(' + str(round(TOTAL_CORRECT*100.0/sum(TOTAL_PER_CLASS),2)) + '%)' 
    
measurePerformance('training_set')
measurePerformance('test_set')


cv2.destroyAllWindows()