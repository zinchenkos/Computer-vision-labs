import time
import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import pickle
import math

import subprocess as sp
import multiprocessing as mp
from os import remove

#akaze = cv2.ORB_create(nfeatures=1000)
akaze = cv2.AKAZE_create()
#akaze = cv2.BRISK_create()

def detect_match(query_img, train_img, min_match_count, keypoints1, descriptors1):
    start_time_3 = time.time()
    train_image = train_img
    keypoints2, descriptors2 = akaze.detectAndCompute(train_image, None)

    end_time_3 = time.time()
    total_processing_time_3 = end_time_3 - start_time_3
    print("Время3: {}".format(total_processing_time_3))
    msed = np.inf
    if not (isinstance(descriptors1, np.float32) & isinstance(descriptors2,
                                                              np.float32)):  
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)


    flann_idx = 1
    index_params = dict(algorithm=flann_idx, trees=5)
    search_params = dict(checks=50)

    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good_matches) > min_match_count:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            return [0], [0], [0], [0], M, [0]

        matchesMask = mask.ravel().tolist()
        try:
            h, w, _ = query_img.shape
        except ValueError:
            h, w = query_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)

        msed = np.mean([np.sqrt(np.sum(diff)) for diff in (np.power(pts - dst, 2))] / (np.sqrt(h ** 2 + w ** 2)))
        # cv2.polylines(train_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        # print( "Not enough matches are found - {}/{}".format(len(good_matches), min_match_count) )
        matchesMask = [0]
        M = None
        dst = None



    return keypoints1, keypoints2, good_matches, matchesMask, M, dst

# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def execute():
        cap = cv2.VideoCapture(2)
        imgTarget = cv2.imread('target.jpg')
        myVid = cv2.VideoCapture('video.mp4')

        #imgTarget = image_resize(imgTarget, height=600)
        #cap = cv2.VideoCapture('IMG_8767.MOV')
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        start = time.time()
        succesqs, imgVideo = myVid.read()

        hT, wT, cT = imgTarget.shape
        imgVideo = cv2.resize(imgVideo,(wT,hT))
        detection = False
        frameCounter = 0
        keypoints1, descriptors1 = akaze.detectAndCompute(imgTarget, None)
        time.sleep(1)
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (frame_width, frame_height))
        k = 0
        total_processing_time = 0
        while(cap.isOpened()):

            success, imgWebcam = cap.read()

            if success == True:
                if k%3==0:
                    start_time = time.time()

                    imgWebcam = image_resize(imgWebcam, height = 800)

                    imgStacked = imgWebcam
                    imgAug = imgWebcam.copy()

                    if detection == False:
                        myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
                        frameCounter = 0
                    else:
                        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
                            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frameCounter = 0
                        success, imgVideo = myVid.read()
                        imgVideo = cv2.resize(imgVideo, (wT, hT))

                    start_time_2 = time.time()
                    kp1, kp2, match, inlier, matrix, dst = detect_match(query_img= imgTarget,
                                                             train_img=imgWebcam,
                                                             min_match_count=10, keypoints1=keypoints1, descriptors1=descriptors1)
                    end_time_2 = time.time()
                    total_processing_time_2 = end_time_2 - start_time_2
                    print("Time_2: {}".format(total_processing_time_2))

                    start_time_4 = time.time()

                    if match == 0:
                        match = np.inf


                    inliers_matches_akaze_test = dict()
                    inliers_matches_akaze_test = [[] for i in range(1)]
                    inliers_matches_akaze_test[0].append((np.sum(inlier) / (len(match))))
                    if (math.isnan(inliers_matches_akaze_test[0][0])):
                                inliers_matches_akaze_test[0][0] = 0.0
                    y_pred = loaded_model.predict(np.array(inliers_matches_akaze_test))

                    end_time_4 = time.time()
                    total_processing_time_4 = end_time_4 - start_time_4
                    print("Time_4: {}".format(total_processing_time_4))
                    start_time_5 = time.time()
                    if y_pred == 1:
                        detection = True
                        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

                        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
                        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
                        maskInv = cv2.bitwise_not(maskNew)
                        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
                        imgAug = cv2.bitwise_or(imgWarp, imgAug)

                        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)
                        imgStacked = stackImages(0.5, ([imgWebcam, imgVideo],[imgWarp, imgAug]))

                        fps = 3 / (total_processing_time)
                    out.write(imgStacked)

                    #cv2.imshow('maskNew', maskNew)
                    #cv2.imshow('maskNew', maskNew)
                    #cv2.imshow('imgWarp', imgWarp)

                    #cv2.imshow('imgFeatures', imgFeatures)
                    #cv2.imshow('Imgtarget', imgTarget)
                    #cv2.imshow('myVid', imgVideo)
                    #cv2.imshow('myWebcam', imgWebcam)

                end_time = time.time()
                if(k%3==0):
                    total_processing_time = total_processing_time + end_time - start_time
                    fps = 3 / (total_processing_time)
                else:
                    total_processing_time =  end_time - start_time

                str = "FPS : %0.1f" % fps

                cv2.putText(imgStacked, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                cv2.imshow('imgStacked', imgStacked)

                cv2.waitKey(1)
                frameCounter +=1
                end_time_5 = time.time()
                total_processing_time_5 = end_time_5 - start_time_5
                print("Time_5: {}".format(total_processing_time_5))
                end_time = time.time()
                total_processing_time = end_time - start_time
                print("Time_1: {}".format(total_processing_time))

            else:
                end_time = time.time()
                total_processing_time = end_time - start_time
                print("Time_1: {}".format(total_processing_time))
                break
        end = time.time()
        seconds = end - start

        print("Time taken : {0} seconds".format(seconds))

        fps = num_frames / seconds
        print("Estimated frames per second : {0}".format(fps))

        cap.release()
        out.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    execute()
