# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the


def nothing(x):
    pass
def save(x):
    if(x==1):
        Thresh1 = np.array([lh,ls,lv])
        Thresh2 = np.array([uh,hs,hv])
        print(Thresh1)
        print(Thresh2)

cv2.namedWindow("HSV")
cv2.createTrackbar("lh", "HSV",0, 179, nothing);
cv2.createTrackbar("ls", "HSV",0, 255, nothing);
cv2.createTrackbar("lv", "HSV",0, 255, nothing);
cv2.createTrackbar("uh", "HSV",179, 179, nothing);
cv2.createTrackbar("hs", "HSV",255, 255, nothing);
cv2.createTrackbar("hv", "HSV",255, 255, nothing);
cv2.createTrackbar("save","HSV",0,1,save);

cv2.setTrackbarPos('lh',"HSV",36)
cv2.setTrackbarPos('ls',"HSV",80)
cv2.setTrackbarPos('lv',"HSV",58)
cv2.setTrackbarPos('uh',"HSV",44)
cv2.setTrackbarPos('hs',"HSV",219)
cv2.setTrackbarPos('hv',"HSV",255)

# list of tracked points
greenLower = (29, 86, 6)
greenLower = (34, 50, 50)
greenUpper = (64, 255, 255)
greenUpper = (42, 255, 255)
#these are actually half of values from something like this
#https://colorpicker.me/#4e6b19

pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=1080)
#	print(frame.shape)
	# Cropping an image
	cropped_frame = frame[80:500, 0:1080]
	frame = cropped_frame	
	blurred = cv2.GaussianBlur(frame, (3, 3), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	#need different color filter.  
	lh = cv2.getTrackbarPos('lh', "HSV")
	ls = cv2.getTrackbarPos('ls', "HSV")
	lv = cv2.getTrackbarPos('lv', "HSV")
	uh = cv2.getTrackbarPos('uh', "HSV")
	hs = cv2.getTrackbarPos('hs', "HSV")
	hv = cv2.getTrackbarPos('hv', "HSV")

	thresh1 = np.array([lh, ls, lv])
	thresh2 = np.array([uh, hs, hv])
	mask = cv2.inRange(hsv, thresh1, thresh2)
	
#	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=1)
	mask = cv2.dilate(mask, None, iterations=1)

#	cv2.imshow("Mask", mask)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid

		for c in cnts:
#		print(len(cnts))
#		c = min(cnts, key=cv2.contourArea)
#		print(c)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > 0:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)

			# update the points queue
			pts.appendleft(center)

			# loop over the set of tracked points
			for i in range(1, len(pts)):
				# if either of the tracked points are None, ignore
				# them
				if pts[i - 1] is None or pts[i] is None:
					continue

				# otherwise, compute the thickness of the line and
				# draw the connecting lines
				thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
				cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
