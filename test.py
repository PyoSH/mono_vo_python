import os
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math
import os

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
        print(lo,mid, hi)
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
dataName="01"

dataset_path = os.path.join('/home/burger/eventVO',dataType ,dataDate, dataName)
image_path = os.path.join(dataset_path, 'sequence_e2vid')
odom_path = os.path.join(dataset_path, 'GT')


### 1. Load lists
image_filenames = [file_path(image_path, f) for f in os.listdir(image_path) if is_image(f)]
image_filenames.sort()
camera_matrix, dist_coef = load_intrinsic(dataset_path)
# print(camera_matrix, camera_matrix.shape)
# print(dist_coef, dist_coef.shape)

### 1.1 load odom & timeStamps
odom = []
odom_fastLio = []
timeStamps_jackal = []
timeStamps_fastLio = []
arr_jackal = np.loadtxt(os.path.join(odom_path, 'jackal_odom_TS_origin.txt'),delimiter=',')
arr_fastLio = np.loadtxt(os.path.join(odom_path, 'fast_lio_TS_origin.txt'),delimiter=',')
timeStamp_seq = list(np.loadtxt(os.path.join(image_path, "timestamps.txt"),delimiter=','))

start = time.time()

for i in range(arr_jackal.shape[0]):
    T = np.zeros(shape=(12, 1) ) 
    for j in range(12):
        T[j,0]= arr_jackal[i, j]
    
    odom.append(T.reshape(3,4) )
    timeStamps_jackal.append(arr_jackal[i,-1])

for i_ in range(arr_fastLio.shape[0]):
    T_ = np.zeros(shape=(12, 1) ) 
    for j_ in range(12):
        T_[j_,0]= arr_fastLio[i_, j_]
    
    odom_fastLio.append(T_.reshape(3,4) )
    timeStamps_fastLio.append(arr_fastLio[i,-1])



# print(len(odom), len(timeStamps_jackal))
detector = cv2.GFTTDetector_create()


### 2. monoVO_python script
lk_params = dict(winSize=(21, 21),
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 30, 0.03))

scale_thres = 0.03 # 0.03
current_pos = np.zeros((3, 1))
current_rot = np.eye(3)

# create graph.
position_figure = plt.figure()
position_axes = position_figure.add_subplot(1, 1, 1)
error_figure = plt.figure()
rotation_error_axes = error_figure.add_subplot(1, 1, 1)
rotation_error_list = []
frame_index_list = []

position_GT_est = plt.figure()
position_GT_est_axes = position_GT_est.add_subplot(1, 1, 1)

print("{} images found.".format(len(image_filenames)))

prev_image = None
# prev_minIdx =None
prev_minIdx = 0
prev_scale = None

pose_list=[]

valid_ground_truth = False
if odom is not None:
    valid_ground_truth = True

for index in range(0,len(image_filenames)):
    # load image
    image = cv2.imread(image_filenames[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.undistort(image, camera_matrix, dist_coef)

    # main process
    keypoint = detector.detect(image)

    if prev_image is None:
        prev_image = image
        prev_keypoint = keypoint
        continue

    ### goodFeaturesToTrack
    tmp_keypoint=list(map(lambda x: [x.pt], prev_keypoint))
    points = np.array(tmp_keypoint,dtype=np.float32)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,
                                           image, points,
                                           None, **lk_params)

    E, mask = cv2.findEssentialMat(p1, points, camera_matrix,
                                   cv2.RANSAC, 0.999, 1.0, None)

    points, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix)

    scale = 1
    
    curr_seq_TS = timeStamp_seq[index]
    
    # calc scale from ground truth if exists.
    if valid_ground_truth:
        
        
        minIdx = search(timeStamps_jackal, curr_seq_TS)
        minIdx_gt = search(timeStamps_fastLio, curr_seq_TS)
        # minIdx = Bsearch(timeStamps_jackal, curr_seq_TS, 0 ,len(timeStamps_jackal))  #  O(log n)
        # print(curr_seq_TS,timeStamps_jackal[minIdx])
        ground_truth = odom[minIdx]
        ground_truth_pos = [ground_truth[0, 3], ground_truth[2, 3]]
        
        if prev_minIdx is None or 0:
            prev_minIdx = minIdx
            continue
        # print(prev_minIdx, minIdx)
        
        previous_ground_truth = odom[prev_minIdx]
        # previous_ground_truth = odom[]
        
        previous_ground_truth_pos = [
            previous_ground_truth[0, 3],
            previous_ground_truth[2, 3]]

        scale = calc_euclid_dist(ground_truth_pos,
                                 previous_ground_truth_pos)
        
    if scale < scale_thres:
        continue

    current_pos += current_rot.dot(t) * scale 
    current_rot = R.dot(current_rot)
    # print(index, 'th pose | scale: %.8f'%scale,'| seqTS: %.8f'%curr_seq_TS, '| odomTS: %8f'%timeStamps_jackal[minIdx])
    

    # get ground truth if eist.
    if valid_ground_truth:
        ground_truth = odom[minIdx]
        position_axes.scatter(ground_truth[0, 3], # Xc
                              ground_truth[2, 3], # Zc
                              marker='^',
                              c='r')
        
    gt_fastLio = odom_fastLio[minIdx_gt]
    position_axes.scatter(gt_fastLio[0, 3], # Xc
                          gt_fastLio[2, 3], # Zc
                          marker='^',
                          c='g')

    # calc rotation error with ground truth.
    if valid_ground_truth:
        ground_truth = odom[index]
        ground_truth_rotation = ground_truth[0: 3, 0: 3] # ?????????
        r_vec, _ = cv2.Rodrigues(current_rot.dot(ground_truth_rotation.T))
        rotation_error = np.linalg.norm(r_vec)
        frame_index_list.append(index)
        rotation_error_list.append(rotation_error)

    position_axes.scatter(current_pos[0][0], current_pos[2][0]) # 이거 왜 3x1? 일반적으로 생각해보면 Xc,Yc, Zc일거고.
    plt.pause(.01)
    
    # print('VO: %.8f'%current_pos[0][0],' %.8f'%current_pos[2][0], '\nGT: %.8f'%ground_truth_pos[0], ' %.8f'%ground_truth_pos[1]) # 순서 맞다.

    img = cv2.drawKeypoints(image, keypoint, None)        

    # cv2.imshow('feature', img)
    # cv2.waitKey(1)

    prev_image = image
    prev_keypoint = keypoint
    prev_minIdx = minIdx
    
    # pose_list.append(current_pos)

# for k in range(len(odom_fastLio)):
#     gt_fastLio=odom_fastLio[k]
#     position_GT_est_axes.scatter(gt_fastLio[0, 3], # Xc
#                               gt_fastLio[2, 3], # Zc
#                               marker='^',
#                               c='r')

# for w in range(len(pose_list)):
#     tmp_pose = pose_list[w]
#     position_GT_est_axes.scatter(tmp_pose[0][0], tmp_pose[2][0])
#     plt.pause(.01)


print('Whole runtime ', time.time()-start)
position_figure.savefig("position_plot.png")
position_GT_est.savefig("gt_est.png")
# rotation_error_axes.bar(frame_index_list, rotation_error_list)
# error_figure.savefig("error.png")
