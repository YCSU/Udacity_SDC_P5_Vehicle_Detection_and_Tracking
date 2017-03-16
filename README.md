# Udacity_SDC_P5_Vehicle_Detection_and_Tracking
Detecting cars on the road

[//]: # (Image References)
[image1]: ./output_images/example_car_1.png
[image2]: ./output_images/example_car_2.png
[image3]: ./output_images/example_car_3.png
[image4]: ./output_images/example_noncar_1.png
[image5]: ./output_images/example_noncar_2.png
[image6]: ./output_images/example_noncar_3.png
[image7]: ./output_images/sliding_windows.jpg
[image8]: ./output_images/detection.jpg
[image9]: ./output_images/heatmap.jpg
[image10]: ./output_images/labels.jpg
[image11]: ./output_images/final.jpg


This is the fifth project for Udacity Self-Driving Car Engineer Nanodegree. For this project, we need to use the sliding-window technique combined with a clssifier to detect cars on the road.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected


### Running the code
The project includes the following files:
* main.py - pipeline for detecting cars. See the comments in the file for explanation  
* utils.py - functions including creating features for classification, generating sliding windows, and drawing boxes around the detected cars
* explore_cspace.py- explore different color spaces with 3D plots
* visualize_hog.py - visualize the hog features
* project_video_output.mp4 - the output video with detected cars
* project_video_output_final.mp4 - implemented with detected lane lines from the fouth project
* README.md - the file reading now

To launch the script, 
```
python main.py
```

## Select features and train a classifier
For training a classifier, we need to decide what features to be fed into the model. I explored the histogram of oriented gradients (HOG) and also different color spaces (in explore_cspace.py and visualize_hog.py). HLS color space is chosen to be the main color space. We use not only the HOG features but also the spatial features and color histograms.

The spatial features is extracted in bin_spatial() (line 13-20 in utils.py). The image is resized to 32x32 and unravel to a one-dimensional vector. The color histograms is extracted in color_hist() (line 23-34 in utils.py). Each channel is binned into 32 bins. The HOG features is extracted in get_hog_features() (line 37-53 in utils.py) with orientations=6, pixels_per_cell=(8, 8) and cells_per_block=(2, 2). These feature are combined (line 56-100 in utils.py) which form a feature vector with length=32x32x3+32x3+7x7x2x2x6=4344.

### Examples of Histogram of Oriented Gradients (HOG), color histograms, and spatial features
The following example uses HLS color space and the parameters mentioned above:

An example of cars.
![][image1]
![][image2]
![][image3]

An example of non-cars.
![][image4]
![][image5]
![][image6]


### Choose a model

## Sliding Window Search
### Choose sliding windows
![][image7]
### detection example
![][image8]

## Video Implementation
Here's a [link to my video result](./project_video_output.mp4) and Here's a [link to my video result](./project_video_output_final.mp4)

### Remove false positivs




![][image9]
![][image10]
![][image11]
