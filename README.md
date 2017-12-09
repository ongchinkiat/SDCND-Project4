## Self-Driving Car Engineer Nanodegree

## Project 4: Advance Lane Finding

## Introduction

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Project Files:

* Notebook File: lanelines.ipynb
* Notebook HTML: lanelines.html
* Project Video: project_video_out.mp4
* Camera Calibration Image: camera-calibration.png
* Undistort Test Image: road-undistort.png
* Thresholding Test Image: threshold-image.png
* Warped View Test Image: warped.png
* Thresholding Warped Test Images: threshold-wraped-1.png , threshold-wraped-2.png
* Sliding Window Test Image: sliding-window.png
* Lane Finding Test Image: lane-find.png

## Camera Calibration

The code for camera calibration is in the "Camera Calibration" section of the lanelines.ipynb Notebook file.

We assume that the coordinates of the chessboard is fixed on the x,y plane, with z=0. We prepare the objp array as a 9x6 grid.

We loop through the 20 calibration images provided, using the cv2.findChessboardCorners to detect the chessboard corners in the images. When found, we append the objpoints array with the objp array, and imgpoints array with the detected points.

With the list of objpoints and imgpoints obtained, we can use the cal_undistort() function to get the camera calibration matrix (mtx) and distortion coefficients (dist).

```
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist
```

To test the parameters obtained, we run the function cal_undistort() for 2 test images, and get this result:

![Camera Calibration Image](https://github.com/ongchinkiat/SDCND-Project4/raw/master/camera-calibration.png "Camera Calibration Image")

## Pipeline

### Step 1: Undistort Image

The first step of our processing pipeline is to undistort the given image. We use the cv2.undistort() function on the Car camera Test images:

![road-undistort](https://github.com/ongchinkiat/SDCND-Project4/raw/master/road-undistort.png "road-undistort")

### Step 2: Color and Gradient Thresholding

The next step is to apply color and gradient thresholding.

For color thresholding, we convert the image to the HLS color space, then apply the threshold values of min = 170, max = 255.

FOr gradient thresholding, we convert the image to gray scale, then apply the Sobel X function with min = 20, max = 100.

The resulting 2 threshold images are then combined into the final threshold image:

![threshold-image](https://github.com/ongchinkiat/SDCND-Project4/raw/master/threshold-image.png "threshold-image")

### Step 3: Perspective Transform

The next step is to apply Perspective Transform to the image, to get a bird's eye view of the lane.

I used the straight_lines1.jpg to estimate the vertices of the lane. Then I arbitrarily choose x = 300 and x = ImageWidth - 300 as the destination vertices.

```
vertices = np.array([[
    (200,imshape[0]),
    (520, 500),
    (763, 500),
    (1110,imshape[0])]], dtype=np.int32)

warp_vertices = np.array([[
  (300,imshape[0]),
  (300, 500),
  (imshape[1]-300, 500),
  (imshape[1]-300,imshape[0])]], dtype=np.int32)

```

The resulted source and destination points are:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 300, 720      |
| 520, 500      | 300, 500      |
| 763, 500      | 980, 500      |
| 1110, 720     | 980, 720      |

We use the cv2.getPerspectiveTransform() function to get the transform parameter M.

Testing it on the straight_lines1.jpg file, we get:

![warped](https://github.com/ongchinkiat/SDCND-Project4/raw/master/warped.png "warped")

### Combining Steps 1 - 3

Steps 1 - 3 are then combined into the function source_img_to_wrap().

```
def source_img_to_wrap(img):
    global mtx
    global dist
    global M
    global imshape
    global thresh_min
    global thresh_max
    global s_thresh_min
    global s_thresh_max
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)

    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    return warped, undist
```

The function is tested with 2 test images:

![threshold-wraped-1](https://github.com/ongchinkiat/SDCND-Project4/raw/master/threshold-wraped-1.png "threshold-wraped-1")

![threshold-wraped-2](https://github.com/ongchinkiat/SDCND-Project4/raw/master/threshold-wraped-2.png "threshold-wraped-2")

### Step 4: Line Finding

The SLiding Window Fit algorithm is used to search for the lane lines in the threshold-warped images.
