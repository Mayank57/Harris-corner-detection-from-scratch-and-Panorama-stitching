# Harris corner detection from scratch and Panorama stitching

## Overview
Panorama stitching refers to the seamless composition of
multiple images with overlapped fields of view. Click a set
of images of any scene from the same position with partial
overlaps. Feature extraction: Create your own implementation
for the Harris corner detector and use the existing OpenCV
implementations for the SIFT detector for feature extraction.
Point Correspondences: Find feature point correspondences
between images using homography. Do image warping. Stitch
the images together. Analysis: Results of image stitching.
Also, comment on the quality of point correspondences obtained by your Harris detector and SIFT.

## Harris Corner Detection
Feature is a piece of information in an image that maintains
its uniqueness even if the orientation of the image changes. Eg:
edge, corners, etc. Harris corner detector is a method to mark
the corners in an image. The idea behind this algorithm is that
wherever there is a corner, there must be a larger variation in
intensity in many directions. If it would have been a case for
an edge then there would not be variation in intensity if we
move in a direction along the edge. Thus keeping the idea of
large variations in mind we need to find little patches of image
( or “windows” ), where such large variations are observed.
There is one disadvantage related to this. If we rescale the
image then accuracy of the algorithm reduces.
It is used to detect corners in an image that can be used as features.

### Algorithm

### Methodology 
Our main task here is to find the difference in intensities of
the windows. In order to do so, we use a small window (i.e.
a matrix of dimension N*N ) and run it all over the image.
We then find the difference in the intensities of the matrices
and determine the score. This score is then compared with a
threshold and we determine whether or not to consider this
point under corner or not.
Explaining the code: In order to determine the change in
intensities here, we use the gradient function. In the gradient
function, we subtract the two neighbouring intensities and
divide them by 2. Thus we obtain a rough change for that
pixel. This change is determined in both the X and Y direction.
Thus from the gradient function, we return dx and dy which
comprises the gradient in the X and Y direction respectively.
Once we have gradient matrices then we tend to find the score
of the small windows one by one. This is done using two for
loops. Sum of each window i.e. the determinant and sum of
primary diagonal i.e. the trace is computed and the score is
determined. If this score is greater than the threshold then we
append the coordinates of the point in a list. At last, in order to
depict these corners, we change the colour of these coordinates
to blue.

## SCALE INVARIANT FEATURE TRANSFORM (SIFT)
SIFT stands for Scale Invariant Feature Transform, it is a
feature extraction method where image content is transformed
into local feature coordinates that are invariant to translation,
scale, and other image transformations. It overcomes the
drawback of the Harris Corner detector method. Now, due
to upscaling more pixels are added to the image matrix which
may increase the number of features is Harris detector is used,
which would not be ideal because features should remain the
same even after some transformation.

### Methodology
SIFT comprises of 4 steps:- Scale-space Extrema Detection Keypoint Localisation Orientation assignment Keypoint
Description
In Scale-space Extrema Detection we apply Gaussian filter
to the input image first. We then scale the image by a sampling
factor of 2 thereby reducing the size and smoothing the image
and then again apply Gaussian filter to the new image. Each
level is known as one octave. This keeps going on until 4
images are produced ( 4 is a researched value ) i.e. 4 octaves.
Sigma value for each octave is different. Consecutive images
of each octave are subtracted to obtain the difference.
Fig. 2.
In Keypoint Localisation, our aim is to detect the maxima
and minima now. This is achieved by comparing a pixel with
its neighbors in the current scale as well as in two adjacent
scales. In short, one pixel will be compared to 26 pixels ( 8
in the current scale and 9 each in the two adjacent scales).
In Orientation Assignment, we compute the magnitude and
direction of the pixels. This is done so that features obtained
can be rotation invariant. We compute the magnitude and
direction of the pixel. Using this, for a key point, we compute
a histogram. This histogram is of 36 bins and each bin is of
width 10 degrees. Now we consider the larger bin. This bin
along with bins with a height of 80
Using the above three methods we were able to compute
location, scale, and orientation. For the descriptor, we need
to use the normalized region around the key point. Thus a
16*16 block is taken and it is further divided into 16 blocks
of 4*4 dimensions. For these 4*4 blocks we generate 8 bin
histograms using magnitude and orientation. Thus the vector
returned through this would be consisting of 128 values.

## FEATURE MATCHING USING HOMOGRAPHY
We were able to retrieve descriptors and key points. Now
as we want to perform panorama on all the images present
we need to match features of consecutive images so that they
can be stitched together. This can be done with the help of
homography. We can perform matching directly but in some
cases some uneven points are matched together, to avoid that
we make use of the Ransac algorithm. As we see in the
above image that even though many points have been correctly
matched still there are many points that are not correctly
matched. Hence we need to remove the parameters which do
not satisfy the model ( known as outliners ) and only keep
points that actually match ( known as inliners ).

### Algorithm
Sample Compute Score Sample means the number of points
that one has to take into consideration and fit into the model.
Compute parameters using these sampled data points. Then
check around the model how many points are satisfying the
model with parameters computed using sampled data points.
The more are the number of data points that have satisfied the
model the greater would be the score for that.

### Methodology
We have found the key points and descriptors using SIFT.
Two descriptors are returned to us. These descriptors are then
matched and k best matches are kept in the list. After this, we
apply Lowe’s ratio test to these remaining k points. If there
is a considerable difference between the two best points ( in
the code we have used 0.75*second best ) then the points are
considered to be good. Here we are comparing the distance and
second best and multiplying it with a factor between 01. Thus
we obtain a list of good matches. Using ransac we are trying
to generate a matrix which will tell us the transformation that
is required to generate image 2 from image 1. Basically the
above mentioned three steps are followed and the model that
fits maximum points with the same transform is the model
selected and this matrix is returned thereby creating an almost
ideal match.

## STITCHING
To stitch 2 images together by using the transfer function (
i.e. the output obtained from Homography ).

### Methdology
Here, we are going to apply a perspective transformation
to one of the images. Basically, a perspective transform may
combine one or more operations like rotation, scale, translation, or shear. The idea is to transform one of the images so
that both images merge as one. To do this, we can use the
OpenCV warpPerspective() function. It takes an image and
the homography as input. Then, it warps the source image to
the destination based on the homography.

## References
* https://docs.opencv.org/master/
* https://aishack.in/tutorials/harris-cornerdetector/
* https://www.pyimagesearch.com/2018/12/17/imagestitching-with-opencv-and-python/
* https://www.youtube.com/watch?v=Cu1f6vpEilg
* https://docs.opencv.org/master/d9/dab/tutorialhomography.html
