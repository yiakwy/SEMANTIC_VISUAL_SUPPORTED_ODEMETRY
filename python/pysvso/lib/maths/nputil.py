'''
Created on 15 Jul, 2019

@author: wangyi
'''

import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import colorsys


# numpy array utils.
# warning: the utilities should accept tensors as input.

def read_img(file_path):
    if not os.path.exists(file_path):
        raise ValueError("Image path [%s] does not exist." % (file_path))
    im = cv2.imread(file_path)
    im = im.astype(np.float32, copy=False)
    # im = cv2.resize(im, (config.HEIGHT, config.WIDTH), interpolation=cv2.INTER_CUBIC)
    return im


def load_images(files, config):
    count = len(files)
    X = np.ndarray((count, config.HEIGHT, config.WIDTH, config.CHANNEL), dtype=np.uint8)
    for i, image_file in enumerate(files):
        image = read_img(image_file)
        X[i] = image
    return X


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def IoU_numeric(left_box, right_box, left_area, right_area):
    # Compute intersection areas
    x1 = max(left_box[0], right_box[0])
    y1 = max(left_box[1], right_box[1])
    x2 = min(left_box[2], right_box[2])
    y2 = min(left_box[3], right_box[3])

    h = max(0, y2 - y1)
    w = max(0, x2 - x1)

    overlap = float(w * h)
    union = left_area + right_area - overlap
    iou = overlap / (union + 1e-6)
    return iou


# Simple implementation of my UIoUs, i.e, universal IoUs
def UIoU_numeric(left_box, right_box, left_area, right_area):
    # Compute intersection areas
    x1 = max(left_box[0], right_box[0])
    y1 = max(left_box[1], right_box[1])
    x2 = min(left_box[2], right_box[2])
    y2 = min(left_box[3], right_box[3])

    h = max(0, y2 - y1)
    w = max(0, x2 - x1)

    overlap = float(w * h)
    union = left_area + right_area - overlap
    iou = overlap / union

    x1 = min(left_box[0], right_box[0])
    y1 = min(left_box[1], right_box[1])
    x2 = max(left_box[2], right_box[2])
    y2 = max(left_box[3], right_box[3])

    h = max(0, y2 - y1)
    w = max(0, x2 - x1)

    chul_area = float(w * h)
    x = 1.0 if abs(iou) < 1e-09 else 0.0
    y = (chul_area - union + overlap) / chul_area
    fixed_chul = (left_box[2] - left_box[0] + right_box[2] - right_box[0]) * \
                 (left_box[3] - left_box[1] + right_box[3] - right_box[1])
    ymin = (fixed_chul - (left_area + right_area)) / float(fixed_chul)
    ymax = 1.0
    uiou = (1 - x) * iou - x * (y - ymin) / (ymax - ymin)
    if uiou > -1 and uiou <= 1:
        return uiou
    else:
        print("left_box", left_box)
        print("right_box", right_box)
        print("iou", iou)
        print("chul_area", chul_area)
        print("x", x)
        print("overlap", overlap)
        print("union", union)
        print("y", y)
        print("fixed_chul", fixed_chul)
        print("ymax", ymax)
        print("ymin", ymin)
        print("uiou", uiou)
        raise Exception("Wrong Value!")


# Non-Max Suppression: simliar to tf.image.non_max_suppression for non-symbolic computation
# see https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
def nms(boxes, scores=None, threshold=0.3):
    """
    @param boxes: np.array with standard tensorflow box order, [x1, y1, x2, y2]
    @param scores: np.array
    @param threshold: float32
    """
    if len(boxes) == 0:
        return 0

    # Compute box area: (x2 - x1) * (y2 - y1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    if scores is not None:
        # Sort boxes indices by box scores
        idx = scores.argsort()[::-1]
    else:
        # see https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
        # Sort boxes indices by bottom-right y-coordinates of bounding box
        idx = np.argsort(boxes[:, 3])

    picked = []

    while len(idx) > 0:
        # Pick one to the list
        i = idx[0]
        picked.append(i)
        # Compute IoU of the picked box with the rest
        ious = np.array([IoU_numeric(boxes[i], boxes[j], area[i], area[j]) for j in idx[1:]])
        remove_idx = np.where(ious > threshold)[0] + 1
        # Remove indices of overlapped boxes
        idx = np.delete(idx, remove_idx)
        idx = np.delete(idx, 0)
    return np.array(picked, dtype=np.int32)


# We will use KL distance to compare two pdf
def logloss(p, q):
    return np.sum(cross_entropy(p, q), axis=0)


def cross_entropy(p, q):
    # print("q(sigmoid): %s" % np.array2string(q, formatter={'float_kind':'{0:6.3f}'.format}))
    # print("p(y): %s" % np.array2string(p, formatter={'float_kind':'{0:6.3f}'.format}))
    # print("entropy: %s" % np.array2string(-p * np.log(q) - (1-p) * np.log(1-q), formatter={'float_kind':'{0:6.3f}'.format}))
    EPILON = 1e-6
    p = np.asarray(p)
    q = np.asarray(q)

    # make it probability distribution
    p = (p - np.min(p)) / (np.max(p) - np.min(p)) * (1 - 1e-6)

    q = (q - np.min(q)) / (np.max(q) - np.min(q)) * (1 - 1e-6)

    return -p * np.log(q + EPILON) - (1 - p) * np.log(1 - q)


def cosine_dist(p, q):
    p = (np.asarray(p) - np.mean(p))
    p = p / (np.linalg.norm(p) + 1e-6)
    q = (np.asarray(q) - np.mean(q))
    q = q / (np.linalg.norm(q) + 1e-6)
    return p.dot(q)