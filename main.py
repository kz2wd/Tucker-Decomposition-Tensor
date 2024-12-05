#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:20:28 2021

Savas, B., & Eldén, L. (2007). Handwritten digit classification using higher order singular value decomposition. Pattern Recognition, 40(3), 993–1003.
 https://doi.org/10.1016/j.patcog.2006.08.004

 Compression and classification
 
@author: coulaud
"""

import h5py
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from math import *
from scipy.ndimage.filters import gaussian_filter

#  tensor class
#   http://tensorly.org/stable/user_guide/
import tensorly as tl
from tensorly import tucker_to_tensor
from tensorly import decomposition as tld
from tensorly import tenalg as tla


def build_full_digits_tensor(X, y, nb_img):
    #
    # Construct the training tensor by digits (class)
    # . Three mode tensor
    # mode 1  image size = number of pixels
    # mode 2  number of images
    # mode 3  categories: 0 1 2 3 4 5 6 7 8 9  
    #
    nb_digits = 10
    nb_val = X.shape[0]
    if len(X.shape) == 4:
        img_size = X[0, :, :].size
    else:
        img_size = X[0, :].size

    shape = [img_size, nb_img, nb_digits]
    digitsTensors = np.zeros(shape, dtype='<f4')
    pos = np.zeros(nb_digits, dtype='int32')

    for i in range(nb_val):
        digit = y[i]
        if pos[digit] == nb_img:
            continue
        if len(X.shape) == 3:
            digitsTensors[:, pos[digit], digit] = X[i, :, :].reshape(img_size)
        else:
            digitsTensors[:, pos[digit], digit] = X[i, :]

        pos[digit] += 1
    #
    # You must complete to get a tensor
    for d in range(nb_digits):
        dmax = pos[d]
        for k in range(dmax, nb_img):
            i = random.randint(0, dmax - 1)
            digitsTensors[:, k, d] = digitsTensors[:, i, d].copy()

    return digitsTensors


def blurring_treatment(X, sigma):
    #
    shape = np.asarray(X.shape)
    new_shape = shape.copy()
    offset = 2
    old_img_size = int(sqrt(shape[0]))
    img_size = old_img_size + 2 * offset
    new_shape[0] = img_size * img_size

    X_blur = np.zeros(new_shape, dtype='<f4')
    for digit in range(shape[2]):
        for i in range(shape[1]):
            img = X[:, i, digit].reshape([old_img_size, old_img_size])
            new_img = np.zeros([img_size, img_size], dtype='<f4')
            new_img[offset:(offset + old_img_size),
            offset:(offset + old_img_size)] = img[:, :]

            blurred = gaussian_filter(new_img, sigma=sigma)
            X_blur[:, i, digit] = blurred.reshape(blurred.size)

    return X_blur


def tucker_hosvd(T, rank, method='svd'):
    '''
        HOSVD algorithm 
    '''

    print("tucker_hosvd rank ", rank, " method ", method)
    d = T.ndim

    factors = []
    sv = []
    for i in range(d):
        flat = tl.unfold(T, i)
        if method == 'svd':
            u, s, v = np.linalg.svd(flat, full_matrices=False)
        elif method == 'eig':
            gram = flat @ flat.T
            print('eig')
            s, u = np.linalg.eig(gram)
            s = s.real
        else:
            exit()
        if (i == d - 1):
            print('MODE', i, s[:rank[i]] / s[0])

        factors.append(u[:, :rank[i]])
        sv.append(s[:rank[i]])
    #            V.append(np.transpose(v)[:, :rank[i]])
    L = [factors[i].T for i in range(d)]
    core = tla.multi_mode_dot(T, L)
    return core, factors, sv


#####################################################################
# .       build_trainig_basis
#####################################################################

def build_trainig_basis(T, rank, K, build_sv=False):
    print("Training ", T.shape)

    basis = []
    factors = []
    core, factors, sv = tucker_hosvd(T, rank)
    F = tla.mode_dot(core, sv[2], 2)

    for mu in range(rank[2]):
        B, _, _ = np.linalg.svd(F, full_matrices=False)
        basis += [np.array(B[0:K])]
    return basis, factors[0].T


def yield_estimator(test_T, y_te, basis, K, Ut):
    # print(test_T.shape)
    for i in range(test_T.shape[1]):
        d = test_T[:,i,:]
        # print(Ut.shape, d.shape)
        dp = Ut @ d
        dp = dp.T
        # print(dp.shape, basis[0].shape)

        residuals = [np.linalg.norm(dp - basis[mu] @ basis[mu].T @ dp) for mu in range(len(basis))]
        print(residuals)
        input()
        estimiation = np.argmin(residuals)
        yield estimiation, estimiation == y_te[i]

####################################################################
# Visualization
#
# https://matplotlib.org/stable/tutorials/introductory/images.html

def main():
    use_usps_data = True
    # use_usps_data = False
    nb_digits = 10
    #
    # MINST dataset handwritten
    # http://yann.lecun.com/exdb/mnist/
    # http://monkeythinkmonkeycode.com/mnist_decoding/ to read
    # the handwritten digits are well written
    # The classification is ease :-)

    #
    # database https://www.kaggle.com/bistaumanga/usps-dataset?select=usps.h5
    #
    # Handwritten Digits USPS dataset.
    #
    # The dataset has 7291 train and 2007 test images. The images are 16*16 grayscale pixels.
    #
    # The dataset is given in hdf5 file format, the hdf5 file has two groups train
    #    and test and each group has two datasets: data and target.
    #
    # The data set is difficult
    # Training set
    #     X_tr the set of images
    #      y_tr the label
    #
    # Read data
    #
    if not use_usps_data:
        file = "minst.h5"
    else:
        file = "usps.h5"

    with h5py.File(file, "r") as hf:
        train = hf.get("train")
        X_tr = train.get("data")[:]
        y_tr = train.get("target")[:]
        test = hf.get("test")
        X_te = test.get("data")[:]
        y_te = test.get("target")[:]

    ##############################################################################
    #
    #  Build the full tensor
    nb_img = np.bincount(y_tr).max()
    digitsTensors = build_full_digits_tensor(X_tr, y_tr, nb_img)
    print("Digit tensor", digitsTensors.shape)

    #
    ##############################################################################
    #
    # Preprocessing - bluring with a gaussian
    #
    # Construct the training tensor by digits (class)
    #
    if use_usps_data:
        sigma = 0.9
        digitsTensors = blurring_treatment(digitsTensors, sigma)
    print(digitsTensors.shape)
    #
    #
    ##############################################################################
    #
    # Training phase:
    print(" training phase")
    K = 10
    p = 64
    q = 64
    rank = [p, q, nb_digits]
    # rank = [p, q, 5]
    basis, projector = build_trainig_basis(digitsTensors, rank, K)
    print(np.shape(basis))

    print(projector.shape)

    #
    ##############################################################################
    #
    # test phase:
    print(" test phase")
    T_test = X_te

    shape = list(T_test.shape)
    print("test shape", shape)
    nb_img = np.bincount(y_te).max()
    test_tensor = build_full_digits_tensor(T_test, y_te, nb_img)
    if use_usps_data:
        sigma = 0.9
        test_tensor = blurring_treatment(test_tensor, sigma)

    shape = list(test_tensor.shape)
    print("new shape : ", shape)
    output = list(yield_estimator(test_tensor, y_te, basis, K, projector))
    print(output)
    print(len(output))
    res = [b for _, b in output]

    print(sum(res) / len(res))

if __name__ == "__main__":
    main()
