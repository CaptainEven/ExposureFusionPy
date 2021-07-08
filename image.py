# encoding=utf-8

import os.path

# from scipy import misc
import imageio
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from sklearn.preprocessing import minmax_scale


def weightedAverage(pixel):
    """
    """
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


def gauss_weight(channel, sigma):
    """
    """
    return np.exp(-(channel - 0.5) ** 2 / (2.0 * sigma ** 2))


def show(color_array):
    """
    Function to show image
    """
    plt.imshow(color_array)
    plt.show()
    plt.axis('off')


def show_gray(gray_array):
    """
    Function to show grayscale image
    """
    fig = plt.figure()
    plt.imshow(gray_array, cmap=plt.cm.Greys_r)
    plt.show()
    plt.axis('off')


class Image(object):
    """Class for Image"""

    def __init__(self, dir_path, path, crop=False, n=0):
        """
        """
        self.path = os.path.join(dir_path, str(path))
        self.dir_path = dir_path

        # ----- Read in image
        # self.array = imageio.imread(self.path)
        # self.array = self.array.astype(np.float32) / 255.0
        self.array = cv2.imread(self.path, cv2.IMREAD_COLOR)
        self.array = self.array / 255.0

        if crop:
            self.crop_image(n)

        self.shape = self.array.shape

    def crop_image(self, n):
        """
        """
        resolution = 2 ** n
        (height, width, _) = self.array.shape
        (max_height, max_width) = (resolution * (height // resolution),
                                   resolution * (width // resolution))
        (begin_height, begin_width) = ((height - max_height) / 2,
                                       (width - max_width) / 2)
        self.array = self.array[int(begin_height):int(max_height + begin_height),
                     int(begin_width):int(max_width + begin_width)]

    @property
    def grayScale(self):
        """
        Grayscale image
        """
        rgb = self.array
        self._grayScale = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return self._grayScale

    def saturation(self):
        """
        Function that returns the Saturation map
        """
        # red_channel = self.array[:, :, 0]
        # green_channel = self.array[:, :, 1]
        # blue_channel = self.array[:, :, 2]
        #
        # mean = (red_channel + green_channel + blue_channel) / 3.0
        # saturation = np.sqrt(((red_channel - mean) ** 2 + (green_channel - mean) ** 2 +
        #                       (blue_channel - mean) ** 2) / 3.0)

        ## @even: using numpy
        saturation = np.std(self.array, axis=2)

        return saturation

    def contrast_old(self):
        """
        Function that returns the Constrast numpy array
        """
        grey = self.grayScale
        contrast = np.zeros((self.shape[0], self.shape[1]))

        ## 这一步可以通过卷积实现加速...
        # padding
        grey_extended = np.zeros((self.shape[0] + 2, self.shape[1] + 2))
        grey_extended[1:self.shape[0] + 1, 1:self.shape[1] + 1] = grey

        # laplace kernel
        #        kernel = np.array([[ -1, -1, -1 ],
        #                           [ -1, 8, -1 ],
        #                            [ -1, -1, -1 ]])
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                contrast[row][col] = np.abs((kernel * grey_extended[row:(row + 3), col:(col + 3)]).sum())

        # normalize
        contrast = (contrast - np.min(contrast))
        contrast = contrast / np.max(contrast)

        return contrast

    def conv2d(self, img, kernel):
        """
        :param img:
        :param kernel:
        :return:
        """
        out_img = np.zeros_like(img)
        if len(img.shape) == 3:
            for channel in range(np.ndim(img)):
                out_img[:, :, channel] = convolve2d(img[:, :, channel], kernel, mode='same')
        else:  # gray img
            out_img = convolve2d(img, kernel, mode='same')

        return out_img

    def contrast(self):
        """
        Function that returns the Constrast numpy array
        """
        grey = self.grayScale

        # ## filter2d 2d convolution
        # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # # contrast = self.conv2d(grey, kernel)
        # contrast = cv2.filter2D(grey, -1, kernel)

        ## Get laplace
        contrast = cv2.Laplacian(grey, -1)

        # normalize(min-max)
        # contrast = (contrast - np.min(contrast))
        # contrast = contrast / np.max(contrast)

        contrast = minmax_scale(contrast)

        return contrast

    def sobel(self):
        """
        Function that returns the Constrast numpy array
        """
        grey = self.grayScale
        sobel_h = np.zeros((self.shape[0], self.shape[1]))
        sobel_v = np.zeros((self.shape[0], self.shape[1]))
        grey_extended = np.zeros((self.shape[0] + 2, self.shape[1] + 2))
        grey_extended[1:self.shape[0] + 1, 1:self.shape[1] + 1] = grey
        kernel1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernel2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                sobel_h[row][col] = np.abs(
                    (kernel1 *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
                sobel_v[row][col] = np.abs(
                    (kernel2 *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
        return sobel_h, sobel_v

    def well_exposedness(self):
        """
        Function that returns the Well-Exposedness map
        """
        b_channel = self.array[:, :, 0]
        g_channel = self.array[:, :, 1]
        r_channel = self.array[:, :, 2]

        sigma = 0.2
        b_exp = gauss_weight(b_channel, sigma)
        g_exp = gauss_weight(g_channel, sigma)
        r_exp = gauss_weight(r_channel, sigma)

        return b_exp * g_exp * r_exp


if __name__ == "__main__":
    im = Image("jpeg", "grandcanal_mean.jpg")
    sat = im.contrast_old()
    show_gray(sat)
