#!/usr/bin/env python3
# coding: utf-8

__author__ = "Pyo Seung Hyun"
__version__ = "1.0.0"

import os
import glob
import numpy as np
from PIL import Image

EXTENSIONS = ['.jpg', '.png']

def file_path(root, filename):
    return os.path.join(root, '{}'.format(filename) )

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def load_image(filename):
    return Image.open(filename)

def load_intrinsic(filename):
    camera_matrix = np.zeros(shape=(9,1))
    dist_coef = np.zeros(shape=(5))

    tmp = np.loadtxt(os.path.join(filename, 'intrinsic.txt'),delimiter=',')
    
    for i in range(len(tmp)):
        # print(i)
        if i < 9:
            camera_matrix[i,0] = tmp[i]
        else:
            dist_coef[i-9] = tmp[i]
    
    camera_matrix =camera_matrix.reshape(3,3)
    dist_coef.reshape(5,1)

    return camera_matrix, dist_coef

class dataSet():
    def __init__(self, type, date, name) -> None:
        self.rootDir = '/home/burger/eventVO'
        self.dataset_path = os.path.join(self.rootDir,type, date, name)
        self.image_path = os.path.join(self.dataset_path, 'sequence_e2vid')
        self.odom_path = os.path.join(self.dataset_path, 'GT')

        self.image_filenames = [file_path(self.image_path, f ) for f in os.listdir(self.image_path) if is_image(f)]
        self.image_filenames.sort()

        self.camera_matrix, self.dist_coef = load_intrinsic(self.dataset_path)

        self.fastLioOdom_path = os.path.join(self.odom_path, 'fast_lio_TS_origin.txt')
        self.jackalOdom_path =  os.path.join(self.odom_path, 'jackal_odom_TS_origin.txt')

        self.odom_j, self.odom_f, self.timeStamp_j, self.timeStamp_f, self.timeStamp_s = self.load_odom_timeStampS()
        
        self.scale_thres = 0.02


    def load_odom_timeStampS(self):
        odom_jackal = []
        odom_fastLio = []
        timeStamps_jackal = []
        timeStamps_fastLio = []
        arr_jackal = np.loadtxt(self.jackalOdom_path,delimiter=',')
        arr_fastLio = np.loadtxt(self.fastLioOdom_path,delimiter=',')
        timeStamp_seq = list(np.loadtxt(os.path.join(self.image_path, "timestamps.txt"),delimiter=','))

        for i in range(arr_jackal.shape[0]):
            T = np.zeros(shape=(12, 1) ) 
            for j in range(12):
                T[j,0]= arr_jackal[i, j]
            
            odom_jackal.append(T.reshape(3,4) )
            timeStamps_jackal.append(arr_jackal[i,-1])

        for i_ in range(arr_fastLio.shape[0]):
            T_ = np.zeros(shape=(12, 1) ) 
            for j_ in range(12):
                T_[j_,0]= arr_fastLio[i_, j_]
            
            odom_fastLio.append(T_.reshape(3,4) )
            timeStamps_fastLio.append(arr_fastLio[i_,-1])

        return odom_jackal, odom_fastLio, timeStamps_jackal, timeStamps_fastLio, timeStamp_seq



        
        