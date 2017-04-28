import errno
import os
import sys

import numpy as np
import cv2

from glob import glob

import time

FOLDER_NAME = ''
IMG_FOLDER = 'images/'
OUT_FOLDER = 'out/'
ALIGNED_FOLDER = '/aligned/'
MASK_FOLDER = '/mask/'
OUTPUT_NAME = "filtered.jpg"
ITERATIONS_FOLDER = '/iterations/'

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

def readImages(image_dir): #from earlier assignment this semester
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(reduce(list.__add__, map(glob, search_paths)))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
              for f in image_files]

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
            .format(image_dir))

    return images

def medianValue(images, i, j):
    vals = []
    for x in range(len(images)):
        vals.append(images[x][i,j])
    vals = np.array(vals)
    blue = []
    red = []
    green = []
    for x in range(len(vals)):
        cur = vals[x]
        blue.append(cur[0])
        green.append(cur[1])
        red.append(cur[2])
    return np.array([np.median(np.array(blue)), np.median(np.array(green)), np.median(np.array(red))], dtype=int)

def medianImage(images, start_time):
    x = images[0].shape[0]
    y = images[0].shape[1]
    medImg = np.ndarray(shape=(x,y,3), dtype=int)
    for i in range(x):
        for j in range(y):
            medImg[i,j] = medianValue(images, i, j)
        status(i, x, start_time)
    return medImg

def transform_image_to_base_image(img1, base):
    MIN_MATCH_COUNT = 20
    # Initiate ORB detector
    orb = cv2.ORB()
    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(base,None)
    # initialize matcher
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfm.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    # filtering matches
    good = matches[0:len(matches)/2]
    # getting source & destination points
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # finding homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    #set shape and create array for points
    h,w,b = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # warping perspective
    orient = np.zeros([img1.shape[0], img1.shape[1], 3], dtype=np.float32)
    for i in range(orient.shape[0]):
        for j in range(orient.shape[1]):
            orient[i, j] = WHITE
    orient = cv2.warpPerspective(orient, M, (w, h))
    #get warped image
    dst = cv2.warpPerspective(img1, M, (w, h))

    return dst, orient

def status(row, height, start_time):
    percent = float(row)/height
    elapsed = (time.time() - start_time)
    print "{0:.0f}%".format(int(percent*100)) + " done | " + str(int(elapsed)) + " seconds elapsed | " + str(int(elapsed/(percent+.0001) - elapsed)) + " seconds remaining"

def checkColor(cur, color):
    return cur[0] == color[0] and cur[1] == color[1] and cur[2] == color[2]

def makeFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame

if __name__ == "__main__":
#GET FOLDER NAME FROM USER
    input_func = None
    try:
        input_func = raw_input
    except NameError:
        input_func = input
    FOLDER_NAME = str(input_func("Enter folder name: "))
    
#CREATE IMAGE PATHS
    IMG_FOLDER = IMG_FOLDER + FOLDER_NAME + '/'
    OUT_FOLDER = IMG_FOLDER + OUT_FOLDER
    ALIGNED_FOLDER = OUT_FOLDER + ALIGNED_FOLDER
    MASK_FOLDER = OUT_FOLDER + MASK_FOLDER
    ITERATIONS_FOLDER = OUT_FOLDER + ITERATIONS_FOLDER

#CREATE DIRECTORY FOR OUTPUT IMAGES AND ALIGNED IMAGES IF THEY DON'T EXIST
    makeFolder(OUT_FOLDER)
    makeFolder(ALIGNED_FOLDER)
    makeFolder(MASK_FOLDER)
    makeFolder(ITERATIONS_FOLDER)

#READ IN IMAGES
    images = readImages(IMG_FOLDER)

#ALIGN IMAGES TO BASE IMAGE
    base = images[0]
    numImages = len(images) - 1
    cv2.imwrite(ALIGNED_FOLDER + '0aligned.jpg', base)
    masks = []
    for img in range(1, len(images)):
        print str("Aligning image " + str(img) + "/" + str(numImages))
        warped, mask = transform_image_to_base_image(images[img], base)
        cv2.imwrite(ALIGNED_FOLDER + str(img) + 'aligned.jpg', warped)
        cv2.imwrite(MASK_FOLDER + str(img) + 'mask.jpg', mask)
        masks.append(mask)
    print ("Aligned!\n")

#COMBINE MASKS
    combMask = masks[0]
    print str("Combining image masks...")
    for x in range(1, len(masks)):
        cur = masks[x]
        print str("Combining mask " + str(x) + "/" + str(len(masks)))
        for i in range(cur.shape[0]):
            for j in range(cur.shape[1]):
                pixel = cur[i,j]
                if pixel[0] < 255 or pixel[1] < 255 or pixel[2] < 255:
                    combMask[i,j] = BLACK
    print ("Combined!\n")
    print str("Saving image mask...")
    cv2.imwrite(MASK_FOLDER + 'mask.jpg', combMask)
    print ("Saved!\n")

 # FILTER TO MAKE MEDIAN IMAGE
    start_time = time.time()
    images = readImages(ALIGNED_FOLDER)
    rawFiltered = medianImage(images, start_time)
    print ("Saving raw filtered image...")
    cv2.imwrite(os.path.join(ITERATIONS_FOLDER, 'raw' + OUTPUT_NAME), rawFiltered)
    print ("Saved!\n")
    maskedFiltered = rawFiltered

# MASK IMAGE
    print ("Masking filtered image...")
    for i in range(combMask.shape[0]):
        for j in range(combMask.shape[1]):
            pixel = combMask[i,j]
            if checkColor(pixel, BLACK):
                maskedFiltered[i,j] = BLACK
    print ("Masked!\n")
    print ("Saving masked filtered image...")
    cv2.imwrite(os.path.join(ITERATIONS_FOLDER, 'masked' + OUTPUT_NAME), maskedFiltered)
    print ("Saved!\n")

    print ("Trimming filtered image...")
    trimmed = trim(maskedFiltered)
    print ("Trimmed!\n")
    print ("Saving trimmed filtered image...")
    cv2.imwrite(os.path.join(ITERATIONS_FOLDER, 'trimmed' + OUTPUT_NAME), trimmed)
    print ("Saved!")

    print ("Cropping top of photo...")
    toprow = 0
    crop = 0
    while crop != 1 and toprow < trimmed.shape[0]:
        count = 0
        for x in range(trimmed.shape[1]): #go across row
            cur = trimmed[toprow][x] #get current pixel
            if checkColor(cur, BLACK): #check if pixel is black
                nxt = trimmed[toprow + 1][x] #get next pixel
                if checkColor(nxt, BLACK):
                    count += 1
            else:
                count += 1
        if count == trimmed.shape[1]:
            crop = 1
        else:
            toprow += 1

    print ("Cropping bottom of photo...")
    bottomrow = trimmed.shape[0] - 1
    crop = 0
    while crop != 1 and bottomrow > 0:
        count = 0
        for x in range(trimmed.shape[1]): #go across row
            cur = trimmed[bottomrow][x] #get current pixel
            if checkColor(cur, BLACK): #check if pixel is black
                nxt = trimmed[bottomrow - 1][x] #get next pixel
                if checkColor(nxt, BLACK):
                    count += 1
            else:
                count += 1
        if count == trimmed.shape[1]:
            crop = 1
        else:
            bottomrow -= 1

    trimmed = trimmed[toprow:bottomrow,0:trimmed.shape[1]]

    print ("Find left side of photo...")
    leftcol = 0
    crop = 0
    while crop != 1 and leftcol < trimmed.shape[1]:
        count = 0
        for x in range(trimmed.shape[0]): #go across row
            cur = trimmed[x][leftcol] #get current pixel
            if checkColor(cur, BLACK): #check if pixel is black
                nxt = trimmed[x][leftcol + 1] #get next pixel
                if checkColor(nxt, BLACK):
                    count += 1
            else:
                count += 1
        if count == trimmed.shape[0]:
            crop = 1
        else:
            leftcol += 1

    print ("Cropping right side of photo...")
    rightcol = trimmed.shape[1] - 1
    crop = 0
    while crop != 1 and rightcol > 0:
        count = 0
        for x in range(trimmed.shape[0]): #go across row
            cur = trimmed[x][rightcol] #get current pixel
            if checkColor(cur, BLACK): #check if pixel is black
                nxt = trimmed[x][rightcol - 1] #get next pixel
                if checkColor(nxt, BLACK):
                    count += 1
            else:
                count += 1
        if count == trimmed.shape[0]:
            crop = 1
        else:
            rightcol -= 1

    trimmed = trimmed[:,leftcol:rightcol]

    print ("Saving cropped filtered image...")
    cv2.imwrite(os.path.join(OUT_FOLDER, 'cropped' + OUTPUT_NAME), trimmed)
    print ("Saved!")
