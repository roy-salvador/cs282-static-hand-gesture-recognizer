#
#	Hand Gesture Recognizer
#	@author Roy Salvador
#
import numpy as np
import argparse
import cv2
import skimage.feature as skf
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib 
 
HOG_WINDOW_SIZE=88
 
# Define Gesture Classes
GESTURE_CLASSES=['A', 'B', 'D', 'E', 'F', 'K', 'L', 'N', 'W', 'Y']
gestureFrame = []
blankFrame = cv2.resize(cv2.imread('model/blank.png'), (290, 290))
i=0
while i < len(GESTURE_CLASSES) :
    gestureFrame.append(cv2.resize(cv2.imread('model/'+ GESTURE_CLASSES[i] + '.png'), (290, 290)))
    i=i+1
gestureClassesFrame = cv2.resize(cv2.imread('model/GestureClasses.png'), (580, 290))
 
# thresholds for HSV skin segmentation
lower = np.array([0, 133, 77], dtype = "uint8")
upper = np.array([255, 173, 127], dtype = "uint8")


# Train MultiClass SVM
#TRAINING_FEATURES = []
#TRAINING_LABELS = []
#trainFile = open('model/hog_training_set_' + str(HOG_WINDOW_SIZE) + '.csv', 'r')
#for line in trainFile:
#    j=0
#    featuresVector = []
#    for value in line.strip().split(',') :
#      if j==0 :
#        TRAINING_LABELS.append(int(value))
#      else :
#        featuresVector.append(float(value)) 
#      j = j+1
#    TRAINING_FEATURES.append(featuresVector)	
#trainFile.close()
#print 'Initializing One vs Rest Multi Class SVM from saved training data'
#OVR_SVM_CLF = OneVsRestClassifier(LinearSVC(random_state=0)).fit(TRAINING_FEATURES, TRAINING_LABELS)

# Load SVM model
OVR_SVM_CLF = joblib.load('model/hog_svm_' + str(HOG_WINDOW_SIZE) + '.pkl')


# capture video from camera
camera = cv2.VideoCapture(0)
 
while True:
    # prepare hand frame
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.rectangle(frame, (100,100), (400,400), (0,0,255), 8, 0)
    handframe = frame[105:395,105:395]
 
 
	# resize the frame, convert it to the YCbCr color space,
	# and determine the pixel intensities that fall into
	# the speicifed upper and lower boundaries
    frame = cv2.resize(frame, (1000, 1000)) 
    converted = cv2.cvtColor(handframe, cv2.COLOR_BGR2YCrCb)
    skinMask = cv2.inRange(converted, lower, upper)
    

 
	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erodedSkinMask = cv2.erode(skinMask, kernel, iterations = 2)
    dilatedSkinMask = cv2.dilate(erodedSkinMask, kernel, iterations = 2)
 
	# blur the mask to help remove noise, then apply the
	# mask to the frame
    blurredSkinMask = cv2.GaussianBlur(dilatedSkinMask, (3, 3), 0)    
    skin = cv2.bitwise_and(handframe, handframe, mask = blurredSkinMask)
        
 
	# show the skin in the image along with the mask
	#np.hstack([skin, graySkin])
    graySkin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    #smallerGray = cv2.resize(graySkin, (HOG_WINDOW_SIZE, HOG_WINDOW_SIZE)) 
    
    # Find contours
    ret,thresh = cv2.threshold(blurredSkinMask,127,255,0)
    _, contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
   
    if len(contours) != 0 :
        cnt = contours[0]
        for c in contours :
            if len(cnt) < len(c) :
                cnt = c
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(skin,(x,y),(x+w,y+h),(0,255,0),2)

        segmentedGraySkin = graySkin[y:y+h,x:x+w]
        resizedSegmentedGraySkin = cv2.resize(segmentedGraySkin, (HOG_WINDOW_SIZE, HOG_WINDOW_SIZE)) 
        
        #hog, hogImage = skf.hog(resizedSegmentedGraySkin, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=True, normalise=True)
        hog = skf.hog(resizedSegmentedGraySkin, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False, normalise=True)
        #cv2.imshow("ResizedImage",  resizedSegmentedGraySkin)
        #cv2.imshow("HogImage", hogImage)
       
       
        temp = np.array(hog).reshape((1,len(hog)))
        predictionFrame = gestureFrame[OVR_SVM_CLF.predict(temp)[0]]
    else :
        predictionFrame = blankFrame
    
    # Update frames
    #cv2.imshow("Original", handframe)
    #cv2.imshow("Converted", converted)
    #cv2.imshow("Skin Pipeline", np.hstack([skinMask, erodedSkinMask, dilatedSkinMask, blurredSkinMask]))
    #cv2.imshow("Result",skin)
    cv2.imshow("Video Capture", frame)
    cv2.imshow("Hand Gesture Recognizer", np.vstack([np.hstack([skin,predictionFrame]), gestureClassesFrame]))
 
	# if the 'q' key is pressed, stop the loop
    k = cv2.waitKey(1)
    if k == ord('q'):
        print 'Quitting Program'
        break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
