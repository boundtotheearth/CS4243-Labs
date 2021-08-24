import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: image preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return gray_image: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    ###Your code here###
    R, G, B = 0.299, 0.587, 0.114

    shape_tuple = (img.shape[0], img.shape[1],)
    img_gray = np.empty(shape_tuple, int)

    for i in range(len(img_gray)):
        for j in range(len(img_gray[i])):
            red = img[i][j][0]
            blue = img[i][j][1]
            green = img[i][j][2]
            img_gray[i][j] = R * red + B * blue + G * green
    ###
    return img_gray


def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    
    ###Your code here####
    # img_filtered_h = img
    # img_filtered_v = img
    # img_filtered_d1 = img
    # img_filtered_d2 = img

    # cv2.filter2D(img, cv2.CV_8U, sobelh, dst=img_filtered_h)
    # cv2.filter2D(img, cv2.CV_8U, sobelv, dst=img_filtered_v)
    # cv2.filter2D(img, cv2.CV_8U, sobeld1, dst=img_filtered_d1)
    # cv2.filter2D(img, cv2.CV_8U, sobeld2, dst=img_filtered_d2)

    # img_grad_h = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
    # img_grad_v = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
    # img_grad_d1 = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize = 5)
    # img_grad_d2 = cv2.Sobel(img, cv2.CV_64F, 1, -1, ksize = 5)
    shape_tuple = (img.shape[0], img.shape[1],)
    img_grad_h = np.empty(shape_tuple, float)
    img_grad_v = np.empty(shape_tuple, float)
    img_grad_d1 = np.empty(shape_tuple, float)
    img_grad_d2 = np.empty(shape_tuple, float)

    # Sobel Filter
    for i in range(len(img_grad_h)):
        for j in range(len(img_grad_h[i])):
            gradient_h = 0
            gradient_v = 0
            gradient_d1 = 0
            gradient_d2 = 0
            for k in range(3):
                for l in range(3):
                    x_index = i + k - 1
                    y_index = j + l - 1

                    if i <= 0 or i == len(img_grad_h) - 1 or j <= 0 or j == len(img_grad_h[i]) - 1:
                        continue

                    # Filter for h
                    gradient_h += img[x_index][y_index] * sobelh[k][l]

                    # Filter for v
                    gradient_v += img[x_index][y_index] * sobelv[k][l]

                    # Filter for d1
                    gradient_d1 += img[x_index][y_index] * sobeld1[k][l]

                    # Filter for d2
                    gradient_d2 += img[x_index][y_index] * sobeld2[k][l]

            img_grad_h[i][j] = abs(gradient_h)
            img_grad_v[i][j] = abs(gradient_v)
            img_grad_d1[i][j] = abs(gradient_d1)
            img_grad_d2[i][j] = abs(gradient_d2)
    ###
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    ###Your code here###
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_pad[pad_width_bef + i][pad_height_bef + j] = img[i][j]
    ###
    return img_pad




##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the convolution operation in a naive 6 nested for-loops. 
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    return response




def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    ###Your code here###
    ###
    return res

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    ###
    return response






###############################################
"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_squares(response, img_ori=None, rec_shape=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()
    H, W = response.shape[:2]
    if rec_shape is None:
        h_rec, w_rec = 25, 25
    else:
        h_rec, w_rec = rec_shape

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.rectangle(response, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (255, 0, 0), 2)
        if img_ori is not None:
            img_ori = cv2.rectangle(img_ori, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (0, 255, 0), 2)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)

## Delete After ##
# data_dir = 'inputs'
# filename = 'wallpaper.jpg'
# img = read_img(os.path.join(data_dir, filename))
# # gray_img = rgb2gray(img)
# # grad_img_h, grad_img_v, grad_img_d1, grad_img_d2 = gray2grad(gray_img)
# grad_img_d2 = pad_zeros(img, 5, 5, 5, 5)
# imgplot = plt.imshow(grad_img_d2, cmap='gray', vmin=0, vmax=255)
# plt.show()
