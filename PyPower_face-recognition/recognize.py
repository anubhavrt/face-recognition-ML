# USAGE

#python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle --image images/test1.jpg

# here we have imported the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

#now her we have  constructed the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="here shows path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="here shows path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="here shows path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="here shows path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="here shows path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="here shows minimum probability to filter weak detections")
args = vars(ap.parse_args())

# nnow we have loaded our serialized face detector from disk
print("[INFO] here loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#then we  load our serialized face embedding model from disk
print("[INFO] now loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# now load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# here we have to load the image, resize it to have a width of 600 pixels (while
# for maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# now lets construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# AND HERE WE HAVE to apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

# now we loop over the detections
for i in range(0, detections.shape[2]):
	#now we need to extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# here we have filtered out weak detections
	if confidence > args["confidence"]:
		# now we need to  compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# now extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# niw need to ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		#here we need to construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# now we need to perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# now need to draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)

# here we show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
