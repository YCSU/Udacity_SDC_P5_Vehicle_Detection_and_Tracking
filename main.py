# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

import pickle
from collections import deque

from utils import slide_window, extract_single_image, draw_boxes, draw_labeled_bboxes



class heatmap():
    def __init__(self, img_shape, threshold=1, deque_len=5):
        self.queue = deque(maxlen=deque_len)
        self.image_queue = deque(maxlen=deque_len)
        self.threshold = threshold
        self.hm_shape = img_shape

    def add_heat(self, bbox_list):
        '''
        Add heatmap from each frame to deque
        '''
        hm = np.zeros(self.hm_shape)
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            hm[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        self.queue.append(hm)

    def aggregate(self):
        '''
        Aggreate the heatmaps from multiple frames
        '''
        if len(self.queue) > 0:
            # Aggregate heatmaps
            agg_hm = np.sum(self.queue, axis=0)
            # Remove false positive
            agg_hm[agg_hm <= self.threshold] = 0
            return agg_hm
        else:
            return np.zeros(self.hm_shape)


def set_windows(img):
    '''
    Set sliding windows
    '''
    windows_list = []
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[384, 640],
                  xy_window=(160, 128), xy_overlap=(0.5, 0.5))
    windows_list.extend(windows)
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[416, 560],
                 xy_window=(128, 96), xy_overlap=(0.5, 0.5))
    windows_list.extend(windows)
    windows = slide_window(img, x_start_stop=[256, 1024], y_start_stop=[410, 570],
                  xy_window=(96, 64), xy_overlap=(0.75, 0.5))
    windows_list.extend(windows)
    windows = slide_window(img, x_start_stop=[320, 960], y_start_stop=[416, 512],
                 xy_window=(64, 48), xy_overlap=(0., 0.5))
    windows_list.extend(windows)

    return windows_list


def search_windows(img, windows, clf, scaler,
                    color_space='HLS',spatial_size=(32, 32),
                    hist_bins=32, hist_range=(0, 256),
                    hog_channel=1, orient=6,
                    pix_per_cell=8, cell_per_block=2):
    '''
    Find windows containg cars
    '''
    # Create an empty list to receive positive detection windows
    on_windows = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # Extract features for that window using single_img_features()
        features = extract_single_image(test_img,
                                    cspace=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_range=hist_range,
                                    hog_channel=hog_channel, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows


def draw_rect_arounbd_cars(img, windows_list, clf, scaler, hm):
    '''
    Pipeline for detecting cars in the video
    '''
    # Candidate of windows containg cars
    windows_list = search_windows(img, windows_list, clf, scaler,
                                  color_space='HLS', spatial_size=(32, 32),
                                  hist_bins=32, hist_range=(0, 256),
                                  hog_channel=1, orient=6,
                                  pix_per_cell=8, cell_per_block=2)
    # Collect heatmaps
    hm.add_heat(windows_list)
    hm.image_queue.append(img)
    agg_hm = hm.aggregate()
    # Label windows for cars
    labels = label(agg_hm)

    cv2.putText(img, "detected {} cars".format(labels[1]),
                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255,0), 2)
    return draw_labeled_bboxes(img, labels)



if __name__ == "__main__":
    with open("scaler.pkl", "br") as f:
        scaler = pickle.load(f)

    with open("svm_model.pkl", "br") as f:
        clf = pickle.load(f)

    # Create window list
    img = plt.imread("test_images/test2.jpg")
    windows_list = set_windows(img)
    print(len(windows_list))
    plt.figure(figsize=(5,3))
    plt.imshow(draw_boxes(img, windows_list))
    plt.savefig("output_images/sliding_windows.jpg", dpi=100)
    plt.tight_layout()

    # Detection example
    plt.figure(figsize=(10,4))
    hm = heatmap(img_shape = (720, 1280), threshold=1, deque_len=1)
    for i in range(1,7):
        img = plt.imread("test_images/test{}.jpg".format(i))
        plt.subplot(2,3,i)
        plt.imshow(draw_rect_arounbd_cars(img, windows_list, clf, scaler, hm))
    plt.savefig("output_images/detection.jpg", dpi=150)
    plt.tight_layout()

    # Create processed project video
    hm = heatmap(img_shape = (720, 1280), threshold=8, deque_len=8)
    output = 'project_video_output.mp4'
    clip2 = VideoFileClip('project_video.mp4')
    draw = lambda x: draw_rect_arounbd_cars(x, windows_list, clf, scaler, hm)
    clip = clip2.fl_image(draw)
    clip.write_videofile(output, audio=False)

    # Create processed test video
    hm = heatmap(img_shape = (720, 1280), threshold=8, deque_len=8)
    output = 'test_video_output.mp4'
    clip2 = VideoFileClip('test_video.mp4')
    draw = lambda x: draw_rect_arounbd_cars(x, windows_list, clf, scaler, hm)
    clip = clip2.fl_image(draw)
    clip.write_videofile(output, audio=False)

    # Plot heatmap for the last 8 frames
    plt.figure(figsize=(16,12))
    for i in range(0,8):
        plt.subplot(4,4,i+1)
        plt.imshow(hm.image_queue[i])
        plt.subplot(4,4,i+9)
        plt.imshow(hm.queue[i], cmap="hot")
    plt.savefig("output_images/heatmap.jpg", dpi=100)
    plt.tight_layout()

    # Plot label figure
    agg_hm = hm.aggregate()
    labels = label(agg_hm)
    plt.figure(figsize=(5,3))
    plt.imshow(labels[0], cmap='gray')
    plt.savefig("output_images/labels.jpg", dpi=100)
    plt.tight_layout()

    # Plot the detection result for the last frame
    plt.figure(figsize=(5,3))
    plt.imshow(draw_labeled_bboxes(hm.image_queue[-1], labels))
    plt.savefig("output_images/final.jpg", dpi=100)
    plt.tight_layout()
