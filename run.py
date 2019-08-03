from __future__ import print_function, unicode_literals

import os
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand_3d

HEIGHT, WIDTH = 240, 320

def plot_hand(coords_hw, axis, color_fixed=None,  markersize=1, alpha=1.0, fillstyle='full'):
    """ Plots a hand stick figure into a matplotlib figure. """
    # define connections and colors of the bones
    bones = [
             (0, 4), (4, 3), (3, 2), (2, 1), 
             (0, 8), (8, 7), (7, 6), (6, 5), 
             (0, 12), (12, 11), (11, 10), (10, 9), 
             (0, 16), (16, 15), (15, 14), (14, 13),
             (0, 20), (20, 19), (19, 18), (18, 17)
            ]

    for connection in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        axis.plot(coords[:, 1], coords[:, 0], color_fixed,  markersize=markersize, \
                  alpha=alpha, fillstyle=fillstyle)

def _main():
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_image', required=True,
                        help='Input image file.')

    args = parser.parse_args()
    img_name = args.input_image

    if not os.path.exists(img_name):
        raise Exception('an image `{}` does not exist'.format(args.input_image))

     # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, HEIGHT, WIDTH, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    image_raw = scipy.misc.imread(img_name)
    image_raw = scipy.misc.imresize(image_raw, (HEIGHT, WIDTH))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v, \
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run(
                                  [
                                    hand_scoremap_tf, image_crop_tf, 
                                    scale_tf, center_tf,
                                    keypoints_scoremap_tf,
                                    keypoint_coord3d_tf
                                  ],
                                  feed_dict={image_tf: image_v}
                                 )

    hand_scoremap_v = np.squeeze(hand_scoremap_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
    
    # post processing
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
    
    #output coordinates
    print('Key points:')  #do inverse transformation of coordinates
    print('|{:<4s}|{:<8s}|{:<8s}|{:<8s}|'.format('№', 'x', 'y', 'z'))
    for idx, point in enumerate(keypoint_coord3d_v):
        print('|{:<4}|{:< 8.4f}|{:< 8.4f}|{:< 8.4f}|'.format(idx, *point))
    
    # visualize
    fig = plt.figure(1)
    
    ax1 = fig.add_subplot(221)
    ax1.imshow(image_raw) 
    plot_hand(coord_hw, ax1, color_fixed='ro', markersize=5)
    ax1.set_title('Key Points')
    
    ax2 = fig.add_subplot(222)
    max_points_scoremap = np.argmax(hand_scoremap_v, 2)
    hand_scoremap_v = max_points_scoremap - np.min(max_points_scoremap)
    hand_area = hand_scoremap_v > 0.0
    
    image_raw[hand_area, 1] = 255.0
    ax2.imshow(image_raw)   
    ax2.set_title('Painted Hand')

    plt.show()
    
if __name__ == '__main__':
    _main()
