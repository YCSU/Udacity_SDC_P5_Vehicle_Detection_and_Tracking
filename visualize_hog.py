import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
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

#img0 = plt.imread("/home/ycsu/SDC/p5_Vehicle_Detection_and_Tracking/CarND-Vehicle-Detection/data/vehicles/KITTI_extracted/4.png")
img0 = plt.imread("/home/ycsu/SDC/p5_Vehicle_Detection_and_Tracking/CarND-Vehicle-Detection/data/non-vehicles/Extras/extra215.png")

plt.figure(figsize=(7.5,2.5))
plt.subplot(121)
plt.imshow(img0)
plt.title("example non-vehicle")
plt.subplot(122)
plt.title("bin spatial")
plt.imshow(cv2.resize(img0, (32,32)))
plt.savefig("CarND-Vehicle-Detection/output_images/example_noncar_1.png", dpi=100)

img = cv2.cvtColor(img0, cv2.COLOR_RGB2HLS)
plt.figure(figsize=(7.5, 2.5))
c0 = img[:,:,0]
c1 = img[:,:,1]
c2 = img[:,:,2]
plt.subplot(131)
plt.title("H channel")
plt.imshow(c0, cmap="gray")
plt.subplot(132)
plt.title("L channel")
plt.imshow(c1, cmap="gray")
plt.subplot(133)
plt.title("S channel")
plt.imshow(c2, cmap="gray")
plt.savefig("CarND-Vehicle-Detection/output_images/example_noncar_2.png", dpi=100)

features0, hog_img0 = hog(c0, 6, pixels_per_cell=(8,8), cells_per_block=(2,2), visualise=True, feature_vector=False)
features1, hog_img1 = hog(c1, 6, pixels_per_cell=(8,8), cells_per_block=(2,2), visualise=True, feature_vector=False)
features2, hog_img2 = hog(c2, 6, pixels_per_cell=(8,8), cells_per_block=(2,2), visualise=True, feature_vector=False)
plt.figure(figsize=(7.5, 2.5))
plt.subplot(131)
plt.title("HOG - H channel")
plt.imshow(hog_img0, cmap="gray")
plt.subplot(132)
plt.title("HOG - L channel")
plt.imshow(hog_img1, cmap="gray")
plt.subplot(133)
plt.title("HOG - S channel")
plt.imshow(hog_img2, cmap="gray")
plt.savefig("CarND-Vehicle-Detection/output_images/example_noncar_3.png", dpi=100)

