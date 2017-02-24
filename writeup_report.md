# Advanced Lane Finding

---

## Camera Calibration

To calibrate the camera, I read in the calibration chessboard files provided in the repo, created object and image points from the images and then fed everything into the cv2.findChessboardCorners function. Here's an example of an undistorted chessboard image:

![Undistorted Chessboard](assets/chessboard_undistorted.png)

---

## Pipeline

#### Distortion Correction

The first step in the pipeline was to undistort the images using the camera calibration in the previous step. Here's an example of an undistorted image. 

![Undistorted Image](assets/image_undistorted.png)

---

#### Thresholding

Next up, the a binary image was produced that thresholded the lane lines. The thresholding code is reproduced below:

```py
def threshold_image(image):
    red = image[:,:,0]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    saturation = hls[:,:,2]
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    
    
    binary = np.zeros_like(gray)
    search = ((scaled_sobel_mag > 120) | (saturation > 120)) & (red > 80)
    binary[search] = 1
    
    return binary
``` 
    
And here's an example of a binary thresholded image:

![Thresholded](assets/image_thresholded.png)
    
---

#### Perspective Transform

The next step was to transform the perspective of the image. I chose the first of the 2 straight test images, and drew a trapezoid along the lane lines. Initially, I drew the trapezoid to the very edge of the field vision, but that caused the transform to stretch at too much at the top, so I scaled back to a point where the stretching wasn't unreasonable. The coordinates and code I used to get the transform matrix were as follows:

```py
src = np.float32([[275, 680], [595, 450], [695, 450], [1050, 680]])
dst = np.float32([[300, 720], [300, 0], [980, 0], [980, 720]])
M = cv2.getPerspectiveTransform(src, dst)
```

Then in order to actually warp an image, I used cv2.warpPerspective with the transform matrix

```
warped = cv2.warpPerspective(image, M, (1280, 720))
```

Here is an example of a warped image:

![Warped](assets/image_warped.png)

---

#### Polynomial Fitting

Once a bird's eye image was available, the left and right lane lines had to be fit seperately. I used the sliding windows approach to group lane lines into left and right bins, then fit each one seperately. To identify the start of the lane lines, I look for the maximum left and right column activation in the binary images along the bottom 220 pixels of the iamge. Then I split the image into 10 vertical windows, starting from the bottom moving up, and looked for activated pixels in a 300x72 box around the midpoint. After each iteration, I reset the midpoint to the mean of the activation locations from the previous box.

The function for the sliding windows and polyfit is reproduced here:

```py
def polyfit(warped):
    # Use peaks of column activations from y=500 onward (bottom 220 pixels) to determine lane start points
    colsums = np.sum(warped[500:], axis=0)
    midpoint = warped.shape[1] // 2
    left_current = np.argmax(colsums[0:midpoint])
    right_current = np.argmax(colsums[midpoint:]) + midpoint
    
    # Use sliding windows to identify lane points
    nwindows = 10
    window_height = warped.shape[0] // nwindows
    left_points = []
    right_points = []
    margin = 150
    for n in range(nwindows, 0, -1):
        bottom = window_height * n
        top = window_height * (n-1)
        left_box = warped[top:bottom, left_current - margin: left_current + margin]
        right_box = warped[top:bottom, right_current - margin: right_current + margin]
        lefts = np.transpose(np.nonzero(left_box)) + [top, left_current - margin]
        rights = np.transpose(np.nonzero(right_box)) + [top, right_current - margin]
        left_points.append(lefts)
        right_points.append(rights)
        if len(lefts) > 0:
            left_current = int(np.mean(lefts[:,1]))
        if len(rights) > 0:
            right_current = int(np.mean(rights[:,1]))
    left_points = np.concatenate(left_points)
    right_points = np.concatenate(right_points)
    
    # Fit a polynomial to the lane points
    left_poly = np.polyfit(left_points[:,0], left_points[:,1], 2)
    right_poly = np.polyfit(right_points[:,0], right_points[:,1],2)
    
    return left_poly, right_poly, left_points, right_points
```

Here's an example of a left and right fit image:

![Polyfit](assets/image_polyfit.png)

---

#### Radius of Curvature

To calculate the radius of curvature, I first had to transform the image from pixel space to world space by calculating the meters per pixel:

```py
lane_width_pixels = right_calc(720) - left_calc(720)
meters_per_pixel = [30 / 720, 3.7 / lane_width_pixels] # [y,x]
```

I was then able to transform the points from the right and left lanes determined in the previous step into world space and recompute the polynomial with the new coordinates. Once I had the 2 polynomials, I used the following function to compute the ROC. The 2nd argument, y, was set to 720 in both cases because I wanted to compute the radius of curvature at the bottom of the image where the car was located. 

```py
def radius_curvature(poly, y):
    A = poly[0]
    B = poly[1]
    C = poly[2]
    dx_dy = 2*A*y + B
    d2x_dy2 = 2*A
    return (1 + dx_dy**2)**1.5 / abs(d2x_dy2)
```

While this provided a raw ROC value, the ROC values could be quite noisy, so I added an exponential moving average filter to the calculation to smooth out the noise in the data

```py
roc = .67*roc + .33 * np.mean((roc_left, roc_right))
```

---

#### Position from Center

The position from center was first calculated in pixels then converted to meters using the meters -> pixel conversion in the Radius of Curvature section. The midpoint of the two lane polynomials was assumed to be the center of the lane, and the midpoint of the image was assumed to be the car's location. The difference between these points was used to determine the offset of the car from center

```py
offset_pixels = ((left_calc(720) + right_calc(720)) // 2) - 640
offset_meters = offset_pixels * meters_per_pixel[1]
```

---

#### Frame Output

Here's an image of a final output frame that combines the lane polynomials with the undistorted original image. The polynomials have been transformed back into the original image space so that they can be overlaid on top of the lane in the undistorted image. 

![Frame](assets/image_frame.png)

---

## Processed Video

Link to the [processed video](assets/processed_video.mp4)

---

## Discussion

The pipeline works very well for the test video that was provided, but it breaks down when attempting to run on the challenge video. Currently, the pipeline does not have any error recovery methods, so if for instance no lane points are identified during thresholding, the code errors out instead of recovering gracefully. 

The pipeline has additionally only been tested out on highway conditions with wide open lanes. If the lane lines weren't clearly indicated or the road was obscured by lots of shadow, the thresholding would likely not work. 

The pipeline is optimized for day driving only. I have not tested it, but I do not imagine it would work well in night driving conditions. 


