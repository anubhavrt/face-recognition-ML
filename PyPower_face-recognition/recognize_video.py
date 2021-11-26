# USAGE

#python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle

# here we have to import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# here we have constructed the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="this isn the path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="tells about the path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="tells about the path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="tells about the path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="tells about the minimum probability to filter weak detections")
args = vars(ap.parse_args())

#herer we have loaded our serialized face detector from disk
print("[INFO] here loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# here we have loaded our serialized face embedding model from disk
print("[INFO] here loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

#her we have to load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# here we have to initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
rows = 480
cols = 640

# below code is useed to start the FPS throughput estimator
fps = FPS().start()

#below code is to defne loop over frames from the video file stream
while True:
	# it is used to grab the frame from the threaded video stream
	frame = vs.read()

	# the it  resize the frame to have a width of 600 pixels (while
	# and used for maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=640)
	(h, w) = frame.shape[:2]

	# here we have constructed a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# her ewe have applied OpenCV's deep learning-based face detector to localize
	# the faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# below code defines loop over the detections
	for i in range(0, detections.shape[2]):
		# her we have extracted the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# her we have filtered out weak detections
		if confidence > args["confidence"]:
			# here lets compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# now wse have to extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			#it  ensures the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# now lets construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# here we have to perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			#now lets  draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# here we have  update the FPS counter
	fps.update()
	# now lets show the output frame
	cv2.imshow("Frame", frame)
    
	key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

#here we have to stop stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# now do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
