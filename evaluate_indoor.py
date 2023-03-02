import time
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import dataset_test


def load_image(filename):
    return Image.open(filename)

def calc_euclid_dist(p1, p2):
    a = math.pow((p1[0] - p2[0]), 2.0) + math.pow((p1[1] - p2[1]), 2.0)
    return math.sqrt(a)

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

def klt_run(arg1, arg2, arg3):
    start = time.time()
    currData = dataset_test.dataSet(arg1, arg2, arg3)
    
    detector = cv2.GFTTDetector_create()
    
    ### 2. monoVO_python script
    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 30, 0.03))
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
    
    print("{} images found.".format(len(currData.image_filenames)))
    
    prev_image = None
    # prev_minIdx =None
    prev_minIdx = 0
    
    pose_list=[]
    
    valid_ground_truth = False
    if currData.odom_j is not None:
        valid_ground_truth = True
    
    
    for index in range(0,len(currData.image_filenames)):
        # load image
        image = cv2.imread(currData.image_filenames[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        image = cv2.undistort(image, currData.camera_matrix, currData.dist_coef)
    
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
    
        E, mask = cv2.findEssentialMat(p1, points, currData.camera_matrix,
                                       cv2.RANSAC, 0.999, 1.0, None)
    
        points, R, t, mask = cv2.recoverPose(E, p1, points, currData.camera_matrix)
    
        scale = 1
        
        curr_seq_TS = currData.timeStamp_s[index]
        
        # calc scale from ground truth if exists.
        if valid_ground_truth:
            
            minIdx = search(currData.timeStamp_j, curr_seq_TS)                                 #  O(n^2)
            # minIdx = Bsearch(currData.timeStamp_j, curr_seq_TS, 0 ,len(currData.timeStamp_j))   #  O(log n), bug alert
            ground_truth = currData.odom_j[minIdx]
            ground_truth_pos = [ground_truth[0, 3], ground_truth[2, 3]]
            
            if prev_minIdx is None or 0:
                prev_minIdx = minIdx
                continue
            
            previous_ground_truth = currData.odom_j[prev_minIdx]
            
            previous_ground_truth_pos = [
                previous_ground_truth[0, 3],
                previous_ground_truth[2, 3]]
    
            scale = calc_euclid_dist(ground_truth_pos,
                                     previous_ground_truth_pos)
            
        if scale < currData.scale_thres:
            continue
        else:
            current_pos += current_rot.dot(t) * scale 
            current_rot = R.dot(current_rot)
            # print(index, 'th pose | scale: %.8f'%scale,'| seqTS: %.8f'%curr_seq_TS, '| odomTS: %8f'%currData.timeStamp_j[minIdx])    
            tmp_pos = np.copy(current_pos)
            pose_list.append(tmp_pos)
    
        # get ground truth if eist.
        if valid_ground_truth:
            ground_truth = currData.odom_j[minIdx]
            position_axes.scatter(ground_truth[0, 3], # Xc
                                  ground_truth[2, 3], # Zc
                                  marker='^',
                                  c='r')
    
        # calc rotation error with ground truth.
        if valid_ground_truth:
            ground_truth = currData.odom_j[minIdx]
            ground_truth_rotation = ground_truth[0: 3, 0: 3] # ?????????
            r_vec, _ = cv2.Rodrigues(current_rot.dot(ground_truth_rotation.T))
            rotation_error = np.linalg.norm(r_vec)
            frame_index_list.append(minIdx)
            rotation_error_list.append(rotation_error)
    
        position_axes.scatter(current_pos[0][0], current_pos[2][0])
        plt.pause(.01)
    
        img = cv2.drawKeypoints(image, keypoint, None)        
    
        cv2.imshow('feature', img)
        cv2.waitKey(1)
    
        prev_image = image
        prev_keypoint = keypoint
        prev_minIdx = minIdx
        
    cv2.destroyAllWindows()
    
    print('Whole runtime ', time.time()-start)
    
    for w in range(len(pose_list)):
        tmp_pose = pose_list[w]
        position_GT_est_axes.scatter(tmp_pose[0][0], tmp_pose[2][0])
    
    for k in range(len(currData.odom_f)):
        gt_fastLio=currData.odom_f[k]
        position_GT_est_axes.scatter(gt_fastLio[0, 3], # Xc
                                  gt_fastLio[2, 3], # Zc
                                  marker='^',
                                  c='r')
    
    
    
    # position_figure.savefig("/plots/"+"pos_odom_KLT_"+str(arg2+arg3)+".png")
    position_GT_est.savefig("/home/burger/mono_vo_python/plots/"+"pos_GT_KLT_"+str(arg2+arg3)+".png")
    print('plt saved')
    # rotation_error_axes.bar(frame_index_list, rotation_error_list)
    # error_figure.savefig("error.png")

def gftt_sift_run(arg1, arg2, arg3):

    start = time.time()
    currData = dataset_test.dataSet(arg1, arg2, arg3)
    
    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)
    prev_image = None
    prev_minIdx =None
    
    valid_ground_truth = False
    if currData.odom_j is not None:
        valid_ground_truth = True
    
    # create graph.
    position_figure = plt.figure()
    position_axes = position_figure.add_subplot(1, 1, 1)
    error_figure = plt.figure()
    rotation_error_axes = error_figure.add_subplot(1, 1, 1)
    rotation_error_list = []
    frame_index_list = []
    
    position_GT_est = plt.figure()
    position_GT_est_axes = position_GT_est.add_subplot(1, 1, 1)
    
    ### 2. Get Inliers
    detector = cv2.GFTTDetector_create()
    # descriptor = cv2.xfeatures2d.SIFT_create() # version problem.
    descriptor = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    
    print(len(currData.image_filenames))
    
    pose_list=[]
    
    for index in range(5,len(currData.image_filenames) ):
        
        if index == 0:
            continue
        
    
        prev_image = load_image(currData.image_filenames[index-1])
        prev_image = np.array(prev_image)
        prev_image = cv2.undistort(prev_image, currData.camera_matrix, currData.dist_coef)
    
        image = load_image(currData.image_filenames[index])
        image = np.array(image)
        image = cv2.undistort(image, currData.camera_matrix, currData.dist_coef)
    
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
        
        E, mask = cv2.findEssentialMat(pt1_arr, pt2_arr, currData.camera_matrix, cv2.RANSAC, 0.999, 1.0, None)
        # print(E, E.shape)
        # print(pt1_arr.shape , pt2_arr.shape)
        # print(pt2_arr.shape)
        _, R, t, mask = cv2.recoverPose(E, pt1_arr, pt2_arr, currData.camera_matrix)
    
        scale = 1.0
    
        curr_seq_TS = currData.timeStamp_s[index]
        if valid_ground_truth:
            
            minIdx = search(currData.timeStamp_j, curr_seq_TS)
            # minIdx = Bsearch(currData.timeStamp_j, curr_seq_TS, 0 ,len(currData.timeStamp_j))  #  O(log n)
            
            ground_truth = currData.odom_j[minIdx]
            ground_truth_pos = [ground_truth[0, 3], ground_truth[2, 3]]
    
        if prev_minIdx is None:
            prev_minIdx = minIdx
            continue
        
        previous_ground_truth = currData.odom_j[prev_minIdx]
        
        previous_ground_truth_pos = [
            previous_ground_truth[0, 3],
            previous_ground_truth[2, 3]]
    
        scale = calc_euclid_dist(ground_truth_pos,
                                     previous_ground_truth_pos)
        
        if scale < currData.scale_thres:
            continue
    
        current_pos += current_rot.dot(t) * scale
        current_rot = R.dot(current_rot)
        # print(index, 'th pose | scale: %.8f'%scale,'| seqTS: %.8f'%curr_seq_TS, '| odomTS: %8f'%timeStamps_GT[minIdx])
        tmp_pos = np.copy(current_pos)
        pose_list.append(tmp_pos)
        # print(current_pos)
    
        if valid_ground_truth:
            ground_truth = currData.odom_j[minIdx]
            position_axes.scatter(ground_truth[0, 3], # Xc
                                    ground_truth[2, 3], # Zc
                                    marker='^',
                                    c='r')
            
        # calc rotation error with ground truth.
        if valid_ground_truth:
            ground_truth = currData.odom_j[index]
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
    
    print('Whole runtime ', time.time()-start)
    
    for k in range(len(currData.odom_f)):
        gt_fastLio=currData.odom_f[k]
        position_GT_est_axes.scatter(gt_fastLio[0, 3], # Xc
                                  gt_fastLio[2, 3], # Zc
                                  marker='^',
                                  c='r')
    
    for w in range(len(pose_list)):
        tmp_pose = pose_list[w]
        position_GT_est_axes.scatter(tmp_pose[0][0], -(tmp_pose[2][0]))
    
    # position_figure.savefig("/plots/"+"pos_odom_GFTT+SIFT_"+str(arg2+arg3)+".png")
    position_GT_est.savefig("/home/burger/mono_vo_python/plots/"+"pos_GT_GFTT+SIFT_"+str(arg2+arg3)+".png")
    print('plt saved')
    # rotation_error_axes.bar(frame_index_list, rotation_error_list)
    # error_figure.savefig("error.png")
    

if __name__ == "__main__":
    dataType="indoor"
    # dataDate = "0228"
    # dataName="01"
    
    datasetPath = "/home/burger/eventVO"
    date_list = os.listdir(os.path.join(datasetPath,dataType))
    
    for date in date_list:
        
        name_list = os.listdir(os.path.join(datasetPath, dataType, date))
        dataDate = date
        
        for name in name_list:
            if os.path.isdir(os.path.join(datasetPath, dataType, date,name)):
                dataName = name
                
                print('date: ',date,'| Name: ',name)    
                
                klt_run(dataType, dataDate, dataName)
                # gftt_sift_run(dataType, dataDate, dataName)
                
            else:
                pass