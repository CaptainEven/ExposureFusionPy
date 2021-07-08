# coding: utf-8

import os
import argparse

import image
import laplacianfusion
import naivefusion
import cv2
import numpy as np
import imageio

# Loading the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-l',
                    '--list',
                    dest='names',
                    type=str,
                    default='test17_img_name_list.txt',
                    help='The text file which contains the names of the images')
parser.add_argument('-f',
                    '--folder',
                    type=str,
                    default='./test17/',
                    dest='folder',
                    help='The folder containing the images')
parser.add_argument('--root',
                    type=str,
                    default='./image_set',
                    help='')
parser.add_argument('-hp',
                    '--heightpyr',
                    dest='height_pyr',
                    type=int,
                    default=6,
                    help='The level height of the Laplacian pyramid')
parser.add_argument('-wc',
                    dest='w_c',
                    type=float,
                    default=1.0,
                    help='Exponent of the contrast')
parser.add_argument('-ws',
                    dest='w_s',
                    type=float,
                    default=1.0,
                    help='Exponent of the saturation')
parser.add_argument('-we',
                    dest='w_e',
                    type=float,
                    default=1.0,
                    help='Exponent of the exposedness')


def run():
    """
    """

    args = parser.parse_args()
    opt = vars(args)  # convert to ordinary dict

    root = opt['root']

    # names = [line.rstrip('\n') for line in open(opt['names'])]
    # folder = opt['folder']

    height_pyr = opt['height_pyr']
    w_c = opt['w_c']
    w_s = opt['w_s']
    w_e = opt['w_e']

    for folder_name in os.listdir(root):
        # folder_name = os.path.split(os.path.abspath(folder))[-1]

        folder = root + '/' + folder_name
        names = [x for x in os.listdir(folder)
                 if x.endswith('.jpg') or x.endswith('.png')]

        res_path = './res/' + folder_name
        if not os.path.isdir(res_path):
            os.makedirs(res_path)

        # # ----- Naive Fusion
        # W = naivefusion.WeightsMap(folder, names)
        # res_naive = W.result_exposure(w_c, w_s, w_e)
        # # res_naive = res_naive.astype(np.uint8)
        # # print(type(res_naive))
        # # image.show(res_naive)
        #
        # res_naive_path = res_path + '/naive_res.jpg'
        # # imageio.imwrite(res_naive_path, res_naive)
        #
        # res_naive = res_naive * 255.0
        # # res_naive = res_naive[:, :, ::-1]
        # res_naive = res_naive.astype(np.uint8)
        # cv2.imwrite(res_naive_path, res_naive)
        #
        # print('{:s} saved.'.format(res_naive_path))

        # ----- Laplacian Fusion

        lap = laplacianfusion.LaplacianMap(folder, names, n=height_pyr)
        res_lap = lap.result_exposure(w_c, w_s, w_e)

        res_lap_path = res_path + '/laplace_res.jpg'
        # imageio.imwrite(res_lap_path, res_lap)

        res_lap = res_lap * 255.0
        # res_lap = res_lap[:, :, ::-1]  # RGB to BGR
        res_lap = res_lap.astype(np.uint8)
        # image.show(res_lap)

        cv2.imwrite(res_lap_path, res_lap)

        print('{:s} saved.\n'.format(res_lap_path))


if __name__ == '__main__':
    run()
    print('\nAll processed done.')