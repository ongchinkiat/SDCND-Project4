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
* Histogram Test Image: histogram.png
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

The Sliding Window Fit algorithm is used to search for the lane lines in the threshold-warped images.

First, a histogram of bright pixels along the x-axis is computed:

![histogram](https://github.com/ongchinkiat/SDCND-Project4/raw/master/histogram.png "histogram")

Dividing the image into halves, the left and right peak positions are used as the starting points of the Sliding Window search.

The parameters used for the Sliding Window are:

* number of sliding windows, nwindows = 9
* width of the windows +/- margin, margin = 100
* minimum number of pixels found to recenter window, minpix = 50

With these parameters, we obtained the red pixels for the left lane search, and blue pixels for the right lane search:

![sliding-window](https://github.com/ongchinkiat/SDCND-Project4/raw/master/sliding-window.png "sliding-window")

We then do a second order polynomial fit on the 2 sets of lane pixels to get the 2 curves.

```
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

To help us estimate the radius of curvature of the lane, we apply  conversion factors (ym_per_pix, xm_per_pix) to convert from pixels to meters.

```
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
```

### Combining Steps 1 - 4

Combining all the 4 steps, we verify the result using the test images.

We also calculated the radius of curvature of the lane by taking the average of the radius of the left and right lanes.

```
left_curverad = ((1 + (2*left_fit_cr[0]*pipelined_img.shape[0]*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

right_curverad = ((1 + (2*right_fit_cr[0]*pipelined_img.shape[0]*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

curve = (left_curverad + right_curverad) / 2
```

The vehicle position within the lane is also estimated, by measuring the deviation of the center of the detected lane lines with the center of the image.

```
vehical_dev = (((left_bottom + right_bottom) / 2.0) - (pipelined_img.shape[1] / 2.0)) * xm_per_pix
```


![lane-find](https://github.com/ongchinkiat/SDCND-Project4/raw/master/lane-find.png "lane-find")

## Pipeline Video

The process_image() function is used to process the given project video.

To speed up processing, we also implemented a previous_line_fit() function which uses the previously found lane lines as the starting point for the search.

To make the search algorithm more robust, we keep a set of the previous frame parameters and use a counter "miss-frame" to help in our decision.

```
global prev_left_fit
global prev_right_fit
global prev_curve
global prev_vehical_dev
global miss_frame
```

For each new frame, we try to use the previous_line_fit() algorithm first. But if the result Vehicle Position deviates too much (> 0.1m) from the previous frame, we revert to the use of the sliding_window_fit() algorithm.

If the result Vehicle Position deviates by more than 0.2m, we discard the result keep the previous result. But if we miss a consecutive 10 frames (miss_frame > 10), we don't use the previous result anymore, and stick to the result of the sliding_window_fit() algorithm.

The resulting processed video is in the file: project_video_out.mp4

## Discussion

In this project, we have successfully implemented an advance lane finding algorithm to detect lane lines in a video stream.

The algorithm implemented depends highly on the contrast between the lane markings and the road surface.

The algorithm may not work well if there are other lane markings on the lane (e.g. speed signs, direction arrows, etc), or when we encounter steep or 90 degree turns, or when the car is changing lanes.
 
