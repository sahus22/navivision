import cv2
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

# Load stereo images and disparity map
left_image  = cv2.imread("D:/Downloads/Sampler/drivstereo/left/2018-07-10-09-54-03_2018-07-10-10-06-55-366.jpg", cv2.IMREAD_GRAYSCALE)
right_image  = cv2.imread("D:/Downloads/Sampler/drivstereo/right/2018-07-10-09-54-03_2018-07-10-10-06-55-366.jpg", cv2.IMREAD_GRAYSCALE)
# disparity_map = cv2.imread("D:/Downloads/Sampler/drivstereo/disp/2018-07-10-09-54-03_2018-07-10-10-06-55-366.png", cv2.IMREAD_GRAYSCALE)

def ShowDisparity(bSize=5):
    # Initialize the stereo block matching object
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize)

    # Compute the disparity image
    disparity = stereo.compute(left_image, right_image)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))

    # Plot the result
    return disparity

disparity_map = ShowDisparity(bSize=25)
focal_length = 2063.200  # In pixels
# 2063.400
baseline = 0.545  # In meters
depth_image = focal_length * baseline / disparity_map
# u_left, v_left = 200,200
# disp = depth_image[v_left, u_left]
# print(disparity_map)
# print(depth_image.shape)
# depth_m = baseline * focal_length / disp
# print("Distance to point: ", depth_m, " meters")

cv2.imshow('Depth Map', depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
