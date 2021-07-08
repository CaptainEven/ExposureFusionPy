# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:26:15 2016

@author: Rachid & Chaima
"""

import cv2
import numpy as np
from scipy import misc

import image
import utils


#
# def div0( a, b ):
#    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
#    with np.errstate(divide='ignore', invalid='ignore'):
#        c = np.true_divide( a, b )
#        c[ ~ np.isfinite( c )] = 0
#        return c


class LaplacianMap(object):
    """Class for weights attribution with Laplacian Fusion"""

    def __init__(self, dir_path, names, n=3):
        """
        names is a liste of names, fmt is the format of the images
        """
        self.images = []
        for name in names:
            self.images.append(image.Image(dir_path, name, crop=True, n=n))

        self.shape = self.images[0].shape
        self.n_imgs = len(self.images)
        self.n_levels = n

    def get_weights_map(self, w_c, w_s, w_e):
        """
        Return the normalized Weight map
        """
        self.weights = []
        sums = np.zeros((self.shape[0], self.shape[1]))

        for image_name in self.images:
            # contrast = image_name.contrast_old()
            contrast = image_name.contrast()
            saturation = image_name.saturation()
            exposedness = image_name.well_exposedness()
            weight = (contrast ** w_c) * (saturation ** w_s) * (exposedness ** w_e) + 1e-12
            self.weights.append(weight)
            sums = sums + weight

        # for idx in range(self.num_images):
        #     self.weights[idx] = self.weights[idx] / sums

        # @even
        self.weights = np.array(self.weights) / sums

        return list(self.weights)

    def get_gauss_pyramid(self, image, n):
        """
        Return the Gaussian Pyramid of an image
        """
        gauss_pyramids = [image]

        for floor in range(1, n):
            # layer = utils.Reduce(gauss_pyramids[-1], 1)

            ## @even
            layer = cv2.pyrDown(gauss_pyramids[-1])

            gauss_pyramids.append(layer)

        return gauss_pyramids

    def get_gauss_pyramid_weights(self):
        """
        Return the Gaussian Pyramid of the Weight map of all images
        """
        self.weights_pyramid = []

        for idx in range(self.n_imgs):
            gauss_pyramids = self.get_gauss_pyramid(self.weights[idx], self.n_levels)
            self.weights_pyramid.append(gauss_pyramids)

        return self.weights_pyramid

    def get_laplace_pyramid(self, image, n):
        """
        Return the Laplacian Pyramid of an image
        """
        # if len(image.shape) == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gauss_pyramids = self.get_gauss_pyramid(image, n)
        laplace_pyramids = [gauss_pyramids[-1]]

        for level in range(n - 2, -1, -1):
            # expanded = utils.Expand(gauss_pyramids[level + 1], 1)

            # @even
            expanded = cv2.pyrUp(gauss_pyramids[level + 1])

            new_level = gauss_pyramids[level] - expanded
            laplace_pyramids = [new_level] + laplace_pyramids

        return laplace_pyramids

    def get_lap_pyramid_for_imgs(self):
        """
        Return all the Laplacian pyramid for all images
        """
        self.lap_pyramid = []

        # compute laplace pyramid for each image
        for idx in range(self.n_imgs):
            img = self.images[idx].array
            lap_pyramids = self.get_laplace_pyramid(img, self.n_levels)
            self.lap_pyramid.append(lap_pyramids)

        return self.lap_pyramid

    def result_exposure(self, w_c=1, w_s=1, w_e=1):
        """
        """
        "Return the Exposure Fusion image with Laplacian/Gaussian Fusion method"
        print("Compute weights")
        self.get_weights_map(w_c, w_s, w_e)

        print("Compute gauss weight pyramid")
        self.get_gauss_pyramid_weights()

        print("Compute laplace pyramid")
        self.get_lap_pyramid_for_imgs()

        result_pyramid = []
        for l in range(self.n_levels):
            # print('level ', l)
            result_floor = np.zeros(self.lap_pyramid[0][l].shape)

            for idx in range(self.n_imgs):  # process each image of the sequence
                # print('image ', idx)
                # for c in range(3):  # process each channel
                #     result_floor[:, :, c] += self.lap_pyramid[idx][l][:, :, c] * self.weights_pyramid[idx][l]

                ## @even
                result_floor = result_floor + self.lap_pyramid[idx][l] * np.expand_dims(self.weights_pyramid[idx][l],
                                                                                        axis=2)

            result_pyramid.append(result_floor)

        ## Fusion: Get the image from the Laplacian pyramid
        self.result_image = result_pyramid[-1]
        for l in range(self.n_levels - 2, -1, -1):
            # print('floor ', l)

            # self.result_image = result_pyramid[l] + utils.Expand(self.result_image, 1)

            # @even
            self.result_image = result_pyramid[l] + cv2.pyrUp(self.result_image)

        self.result_image[self.result_image < 0] = 0
        self.result_image[self.result_image > 1] = 1

        return self.result_image


if __name__ == "__main__":
    names = [line.rstrip('\n') for line in open('list_images.txt')]
    lap = LaplacianMap('arno', names, n=6)
    res = lap.result_exposure(1, 1, 1)
    image.show(res)
    misc.imsave("res/arno_3.jpg", res)
