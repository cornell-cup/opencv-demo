# OpenCV Contour Overlay Tutorial
# Finds a black square and overlays your own image

# Install Anaconda3 https://www.continuum.io/downloads
import numpy
from scipy.spatial import distance as dist

# pip install opencv-python
import cv2

import math
import sys
import time

# Order contour points
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[numpy.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[numpy.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[numpy.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[numpy.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return numpy.array([tl, tr, br, bl], dtype="float32")


# Overlay image
def drawOverlay(img, overlay):
    # Convert to grayscale
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = numpy.uint8(gimg)

    # Start timing
    total_start = time.time()

    # Compute edges using the Canny edge detector
    eimg = cv2.Canny(gimg, 100, 200)
    # Find contours in the detected edges
    cimg, contours, hierarchy = cv2.findContours(eimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    maxsquare = None
    maxsquarearea = 0
    for contour in contours:
        # Approximate the corners of the polygon
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, perimeter * 0.05, True)
        
        # Ignore non-quadrilaterals
        if len(corners) != 4:
            continue
        
        # Ignore concave shapes
        if not cv2.isContourConvex(corners):
            continue
        
        # Find the quad of largest area
        corners = corners.reshape((corners.shape[0], 2))
        squares.append(corners)
        if area > 400 and area > maxsquarearea:
            maxsquare = corners
            maxsquarearea = area
    
    # Draw outlines on the image
    cv2.polylines(img, squares, True, (255, 255, 0), 1)
    if maxsquare is None:
        return img
    
    cv2.polylines(img, [maxsquare], True, (255, 0, 0), 3)

    # Compute the overlay transform
    ih, iw, _ = img.shape
    oh, ow, _ = overlay.shape
    rect = numpy.array([
        [0, 0],
        [ow-1, 0],
        [ow-1, oh-1],
        [0, oh-1]
    ], dtype="float32")
    # Orient corners
    dst = order_points(maxsquare).astype("float32")
    transform = cv2.getPerspectiveTransform(rect, dst)
    
    # Warp and mask the overlay
    warped = cv2.warpPerspective(overlay, transform, (iw, ih))
    mask = numpy.zeros(img.shape, dtype="uint8") 
    cv2.drawContours(mask, [maxsquare], 0, (255, 255, 255), -1)
    mask = mask.astype("bool")
    
    # Copy the overlay to the original image
    numpy.copyto(img, warped, where=mask)

    total_end = time.time()
    print("Total Time:", total_end - total_start)

    return img


# Main loop
def loop(overlay):
    # Input details
    width = 1280
    height = 720
    fps = 30
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    vc.set(cv2.CAP_PROP_FPS, fps)
    
    # Main GUI loop
    cv2.namedWindow("camera")
    frame = 0
    disp = None
    while True:
        ret, img = vc.read()
        if ret:
            disp = drawOverlay(img, overlay)

        key = cv2.waitKey(1)
        if key == 27:
            # Quit
            break

        frame += 1
        if disp is not None:
            cv2.imshow("camera", disp)

    vc.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    image = cv2.imread("logo_white@256.png")
    loop(image)
