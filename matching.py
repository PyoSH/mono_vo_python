import os
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

EXTENSIONS = ['.jpg', '.png']

def file_path(root, filename):
    return os.path.join(root, '{}'.format(filename) )

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def load_image(filename):
    return Image.open(filename)

def calc_euclid_dist(p1, p2):
    a = math.pow((p1[0] - p2[0]), 2.0) + math.pow((p1[1] - p2[1]), 2.0)
    return math.sqrt(a)

def load_intrinsic(filename):
    camera_matrix = np.zeros(shape=(9,1))
    dist_coef = np.zeros(shape=(5))

    tmp = np.loadtxt(os.path.join(dataset_path, 'intrinsic.txt'),delimiter=',')
    
    for i in range(len(tmp)):
        # print(i)
        if i < 9:
            camera_matrix[i,0] = tmp[i]
        else:
            dist_coef[i-9] = tmp[i]
    
    camera_matrix =camera_matrix.reshape(3,3)
    dist_coef.reshape(5,1)

    return camera_matrix, dist_coef

def Bsearch(arr, tgt, low, high):
    mid = int(low +(high - low)/2)
    lo = int(low)
    hi = int(high) -1
    thres = 0.01
    while(lo<=hi):
        mid = int(lo +(hi - lo)/2)
        # print(lo,mid, hi)
        if abs(arr[mid]-tgt) <= thres:
            return mid
        elif arr[mid] < tgt:
            lo = mid+1
        else:
            hi = mid-1
    return lo

def search(arr, tgt): # O(n^2)

    tmp_distList = []
    for i in range(len(arr)):
        kim = abs(tgt - arr[i])
        tmp_distList.append(kim)

    lee = np.array(tmp_distList,dtype=np.float32)
    minIdx = lee.argmin()

    return minIdx


dataType="indoor"
dataDate = "0228"
dataName="00"

dataset_path = os.path.join('/home/burger/eventVO',dataType ,dataDate, dataName)
image_path = os.path.join(dataset_path, 'sequence_e2vid')
odom_path = os.path.join(dataset_path, 'GT')


### 1. Load lists
image_filenames = [file_path(image_path, f) for f in os.listdir(image_path) if is_image(f)]
image_filenames.sort()
camera_matrix, dist_coef = load_intrinsic(dataset_path)


# load odom & timeStamps
odom = []
timeStamps_GT = []
arr_origin = np.loadtxt(os.path.join(odom_path, 'jackal_odom_TS_origin.txt'),delimiter=',')
timeStamp_seq = list(np.loadtxt(os.path.join(image_path, "timestamps.txt"),delimiter=','))

for i in range(arr_origin.shape[0]):
    T = np.zeros(shape=(12, 1) ) 
    for j in range(12):
        T[j,0]= arr_origin[i, j]
    
    odom.append(T.reshape(3,4) )
    timeStamps_GT.append(arr_origin[i,-1])

### 
current_pos = np.zeros((3, 1))
current_rot = np.eye(3)
prev_image = None
prev_minIdx =None
prev_scale = None

valid_ground_truth = False
if odom is not None:
    valid_ground_truth = True

# create graph.
position_figure = plt.figure()
position_axes = position_figure.add_subplot(1, 1, 1)
error_figure = plt.figure()
rotation_error_axes = error_figure.add_subplot(1, 1, 1)
rotation_error_list = []
frame_index_list = []

### 2. Get Inliers
detector = cv2.GFTTDetector_create()
# descriptor = cv2.xfeatures2d.SIFT_create() # version problem.
descriptor = cv2.SIFT_create()
bf = cv2.BFMatcher()

print(len(image_filenames))

for index in range(3,len(image_filenames) ):
    
    if i == 0:
        continue
    

    prev_image = load_image(image_filenames[index-1])
    prev_image = np.array(prev_image)
    prev_image = cv2.undistort(prev_image, camera_matrix, dist_coef)

    image = load_image(image_filenames[index])
    image = np.array(image)
    image = cv2.undistort(image, camera_matrix, dist_coef)

    kps1  = detector.detect(prev_image,None) 
    kps2  = detector.detect(image,None) 

    kps1, des1 = descriptor.compute(prev_image,kps1)
    kps2, des2 = descriptor.compute(image,kps2)

    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    pt1_list = []
    pt2_list = []
    
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    display = np.concatenate([cv2.drawKeypoints(image, kps2, None), cv2.drawKeypoints(prev_image, kps1, None)], axis=0) 

    for i in range(len(good) ):
        pt1 = kps1[good[i][0].queryIdx].pt
        pt2 = kps2[good[i][0].trainIdx].pt
        pt1_list.append(pt1)
        pt2_list.append(pt2)
        
        pt2 = (pt2[0], pt2[1] + image.shape[0])

        y1 = int(pt1[0]) # rows
        x1 = int(pt1[1]) # column

        y2 = int(pt2[0])
        x2 = int(pt2[1])

        cv2.line(display, (y1, x1), (y2, x2), (0, 255, 0), 1)
        
    pt1_arr = np.array(pt1_list,dtype=np.float32)
    pt2_arr = np.array(pt2_list,dtype=np.float32)
    # print(points.shape)
    
    E, mask = cv2.findEssentialMat(pt1_arr, pt2_arr, camera_matrix, cv2.RANSAC, 0.999, 1.0, None)
    # print(E, E.shape)
    # print(pt1_arr.shape , pt2_arr.shape)
    # print(pt2_arr.shape)
    _, R, t, mask = cv2.recoverPose(E, pt1_arr, pt2_arr, camera_matrix)

    scale = 1.0

    curr_seq_TS = timeStamp_seq[index]
    if valid_ground_truth:
        
        minIdx = search(timeStamps_GT, curr_seq_TS)
        # minIdx = Bsearch(timeStamps_GT, curr_seq_TS, 0 ,len(timeStamps_GT))  #  O(log n)
        
        
        ground_truth = odom[minIdx]
        ground_truth_pos = [ground_truth[0, 3], ground_truth[2, 3]]

    if prev_minIdx is None:
        prev_minIdx = minIdx
        continue
    
    previous_ground_truth = odom[prev_minIdx]
    
    previous_ground_truth_pos = [
        previous_ground_truth[0, 3],
        previous_ground_truth[2, 3]]

    scale = calc_euclid_dist(ground_truth_pos,
                                 previous_ground_truth_pos)
    
    scale_thres = 0.04
    if scale < scale_thres:
        continue

    current_pos += current_rot.dot(t) * scale
    current_rot = R.dot(current_rot)
    print(index, 'th pose | scale: %.8f'%scale,'| seqTS: %.8f'%curr_seq_TS, '| odomTS: %8f'%timeStamps_GT[minIdx])

    if valid_ground_truth:
        ground_truth = odom[minIdx]
        position_axes.scatter(ground_truth[0, 3], # Xc
                                ground_truth[2, 3], # Zc
                                marker='^',
                                c='r')
        
    # calc rotation error with ground truth.
    if valid_ground_truth:
        ground_truth = odom[index]
        ground_truth_rotation = ground_truth[0: 3, 0: 3] # ?????????
        r_vec, _ = cv2.Rodrigues(current_rot.dot(ground_truth_rotation.T))
        rotation_error = np.linalg.norm(r_vec)
        frame_index_list.append(index)
        rotation_error_list.append(rotation_error)

    position_axes.scatter(current_pos[0][0], -(current_pos[2][0]))
    plt.pause(.01)

    prev_minIdx = minIdx
    cv2.imshow('display', display)
    if cv2.waitKey(10) == 27: 
        break
    cv2.waitKey(1)

cv2.destroyAllWindows()

position_figure.savefig("position_plot_GFTT+SIFT.png")
rotation_error_axes.bar(frame_index_list, rotation_error_list)
error_figure.savefig("error.png")