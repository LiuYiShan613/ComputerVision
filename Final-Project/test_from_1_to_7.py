# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:34:14 2021

@author: Johnny
"""
from __future__ import print_function
import cv2
import numpy as np

def alignImages(input_img, ref_img, i, MAX_FEATURES = 2000, GOOD_MATCH_PERCENT = 0.06):

    # Convert images to grayscale
    input_img_Gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    ref_img_Gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(input_img_Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_img_Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    #run "matches = sorted(matches, key=lambda x:x.distance)" if you got error above

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(input_img, keypoints1, ref_img, keypoints2, matches, None)
    cv2.imwrite("match/match"+str(i)+".jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    print("mask:",mask.shape)
    
    # Use homography
    height, width, channels = ref_img.shape
    input_img_Reg = cv2.warpPerspective(input_img, h, (width, height))

    return input_img_Reg, h


for i in range(1,8):

    # Read image to be aligned
    imFilename = "input_img/input_img"+str(i)+".jpg"
    print("Reading image to align : ", imFilename);  
    input_img = cv2.imread(imFilename, cv2.IMREAD_COLOR)    

    # Read reference image
    refFilename = "ref_img/ref_img"+str(i)+".jpg"
    print("Reading reference image : ", refFilename)
    ref_img = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    imReg, h = alignImages(input_img, ref_img, i)

    # Write aligned image to disk. 
    outFilename = "result/result"+str(i)+".jpg"
    print("Saving aligned image : ", outFilename); 
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h,"\n")