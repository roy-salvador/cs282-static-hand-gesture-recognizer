import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('training_set/Y_HAZEL.avi',fourcc, 10.0, (290,290))

# define the upper and lower boundaries of the HSV pixel for skin segmentation
lower = np.array([0, 133, 77], dtype = "uint8")
upper = np.array([255, 173, 127], dtype = "uint8")

is_recording=False

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # grab the current frame
        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame, (100,100), (400,400), (0,0,255), 8, 0)
        handframe = frame[105:395,105:395]
 
 
       # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        frame = cv2.resize(frame, (1000, 1000)) 
        converted = cv2.cvtColor(handframe, cv2.COLOR_BGR2YCrCb)
        skinMask = cv2.inRange(converted, lower, upper)
 
        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(handframe, handframe, mask = skinMask)
 
        # show the skin in the image along with the mask
        #np.hstack([skin, graySkin])
        graySkin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        graySkinBGR =  cv2.cvtColor(graySkin, cv2.COLOR_GRAY2BGR)
        
        # write the flipped frame
        if is_recording :
            print 'Recording.... Press q to stop'
            out.write(graySkinBGR)

        cv2.imshow('Original Frame',frame)
        cv2.imshow('Hand Frame', graySkin)
        
        
        
        k = cv2.waitKey(1)
        print k
        # Start Recording
        if k == ord('s') :
            print 'Now recording...'
            is_recording=True
        # Quit
        elif k == ord('q'):
            print 'q'
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()