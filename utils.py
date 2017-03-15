import cv2
import numpy as np
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pickle



def bin_spatial(img, size=(32, 32)):
    '''
    Compute binned color features
    '''
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Compute color histogram features
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    '''
    HOG features
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def extract_features(imgs, cspace='BGR', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        hog_channel='ALL', orient=8,
                        pix_per_cell=8, cell_per_block=2,
                        vis=False, feature_vec=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'BGR'
        if cspace != 'BGR':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_feature = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() to get color histogram features
        color_feature = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate([spatial_feature, color_feature, hog_features]))

    # Return list of feature vectors
    return features


def extract_single_image(image, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        hog_channel='ALL', orient=8,
                        pix_per_cell=8, cell_per_block=2,
                        vis=False, feature_vec=True):

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    # Apply bin_spatial() to get spatial color features
    spatial_feature = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() to get color histogram features
    color_feature = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)

    return np.concatenate([spatial_feature, color_feature, hog_features])



def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] =img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    xstride = np.int(xy_window[0]*(1.-xy_overlap[0]))
    ystride = np.int(xy_window[1]*(1.-xy_overlap[1]))

    # Compute the number of windows in x/y
    xnum_win = (xspan - xy_window[0])//xstride + 1
    ynum_win = (yspan - xy_window[1])//ystride + 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for xnum in range(xnum_win):
        for ynum in range(ynum_win):
            # Calculate each window position
            x = xnum * xstride + x_start_stop[0]
            y = ynum * ystride + y_start_stop[0]
            # Append window position to list
            window_list.append(((x, y), (x+xy_window[0], y+xy_window[1])))
    # Return the list of windows
    return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 5)
    # Return the image
    return img


def add_heat(image, bbox_list):
    heatmap = np.zeros(image.shape[:-1])
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


if __name__ == "__main__":
    noncars=glob.glob("data/non-vehicles/*/*.png")
    cars=glob.glob("data/vehicles/*/*.png")
    print("# of non-cars: {0}".format(len(noncars)))
    print("# of cars: {0}".format(len(cars)))

    img_paths = np.concatenate([cars, noncars])
    X = extract_features(img_paths, cspace='HLS', spatial_size=(32, 32), hist_bins=32, orient=6, hog_channel=1)
    y = np.concatenate([np.ones(len(cars)), np.zeros(len(noncars))])

    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=42, stratify=y)
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    print("training acc: ", clf.score(X_train, y_train))
    print("testing acc: ", clf.score(X_test, y_test))

    with open("scaler.pkl", "bw") as f:
        pickle.dump(X_scaler, f)

    with open("svm_model.pkl", "bw") as f:
        pickle.dump(clf, f)

    #for ssize in (16, 32):
    #    for hist_bins in (16, 32):
    #        for orient in (6, 8, 10):
    #            for cell_per_block in (2, 4):
    #                X = extract_features(img_paths, cspace='HLS', spatial_size=(ssize, ssize), hist_bins=hist_bins, hog_channel=1, orient=orient, cell_per_block=cell_per_block)
    #                y = np.concatenate([np.ones(len(cars)), np.zeros(len(noncars))])
    #                X_scaler = StandardScaler().fit(X)
    #                scaled_X = X_scaler.transform(X)
     #               X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
     #               clf = LinearSVC()
     #               clf.fit(X_train, y_train)
     #               print("spatial_size={}, hist_bins={}, orient={}, cell_per_block={}".format(ssize, hist_bins, orient, cell_per_block))
     #               print("training acc: ", clf.score(X_train, y_train))
     #               print("testing acc: ", clf.score(X_test, y_test))