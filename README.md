# cs282-static-hand-gesture-recognizer

A python application which recognizes a subset of the American Sign Language (ASL) in real time using Histogram of Oriented Gradients (HOG). A Mini Project requirement for CS 282 course at University of the Philippines Diliman AY 2015-2016 under Sir Prospero Naval.

![Demo Application](https://github.com/roy-salvador/cs282-static-hand-gesture-recognizer/blob/master/Demo.gif)

## Requirements
* [OpenCV](http://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html#gsc.tab=0) Computer Vision Library
* [Scikit Image](http://scikit-image.org/)
* [Scikit Learn](http://scikit-learn.org/)
* Python 2.7.11


## Instructions
1. Clone and download the repository.
2. Save your gestures using the recorder utility. 
  
  ```  
  python handgesture-recorder.py
  ```

3. Generate the HOG features file. This will be used as training examples of the SVM. Place it inside model directory.  
  
  ```  
  python handgesture-featureextractor.py
  ```
  
4. Train a multi class SVM using the HOG features file and see the training and test set performance. This also saves the
SVM model to files named hog_svm*.pkl*. Place them all inside the model directory.
  
  ```  
  python handgesture-measureperformance.py
  ```
  
5. Update the HOG_WINDOW_SIZE parameter with what was used in feature extraction. Run the Recognizer Application.

  ```  
  python handgesture-recognizer.py
  ```
