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

The spatial features are extracted in bin_spatial() (line 13-20 in utils.py). The image is resized to 32x32 and unravels to a one-dimensional vector. The color histograms are extracted in color_hist() (line 23-34 in utils.py). Each channel is binned into 32 bins. The HOG features are extracted in get_hog_features() (line 37-53 in utils.py) with orientations=6, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) for only the L channel. These feature are combined (line 56-100 in utils.py) which form a feature vector with length=32x32x3+32x3+7x7x2x2x6=4344.

### Choose a model and parameters
Due to the large size of features, we use LinearSVC() in scikit-learn to train a classifier (line 227-250 in utils.py). The dataset is split into a training set (90%) and a test set (10%). The parameters for extracting features are determined by running a grid search with cross-validation (line 252-265 in utils.py). The feature vector is normalized by StandardScaler() (line 236 in utils.py).

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



## Sliding Window Search
### Choose sliding windows
To detect cars in the image, we employ a sliding window search (line 48-66 in main.py, line 143-179 in utils.py). The ranges and scales to search are determined by eye-balling the positions and sizes of the cars in the videos. There are totally 173 windows.

![][image7]

### Detection example
The detection pipeline is the following. We resize the image picked by each sliding window to 64x64, extract features, normalize it, and feed it into the classifier (line 69-97 in main.py). To minimize the false positives, we implement the heat map technique introduced in the lecture (line 15-45). First, we define that the pixels winthin the window in which a car is detected are activated. we notice that, for pixels representing a car, they are usually acvtivated several times, so pixels only activated once are regearded as false positives and are ignored (line 34-45 and line 141 in main.py). Using the label() function from scipy.ndimage.measurements, we redraw the boxs around the pixels that are activated more than one time (line 192-205 in utils.py). An example is shown below.
![][image8]

## Video Implementation
Here's [a link to my result for detecting cars in the video](./project_video_output.mp4) and here's [another link to my video result combining lane and car detection](./project_video_output_final.mp4). The pipeline for processing the video is similar to the one for the image. However, to remove the false pistives, we store the heap maps from frame to frame in a queue, and sum up the eight consecutive heat maps in a queue to form a final heat map. Pixels that are activated less than nine times in the fnal heat map are ignord  (line 34-45 and line 150 in main.py). 

### Remove false positivs
Here is an example of removing false positives. The images below are eight consecutive frames and the corresponding heat maps.
![][image9]

By combining the heat maps and ignoring pixels activated less than nine times, the label() function from scipy.ndimage.measurements detects two clusters of pixels.
![][image10]

The final result by drawing the boxes around the region detected.
![][image11]

## Discussion
The current pipeline works in good lighting conditions and clear weather. However, for a dark night or a snowy day, the classifier probably needs to be retrained because it relies heavily on the HOG features generated from the L channel, and those features might not be useful under those conditions.

Another potential improvement is the efficiency of extracting the HOG features. The sub-sampling mentioned in the lectures can be implemented to speed up the processing frame rate.

