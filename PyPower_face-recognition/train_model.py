# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle
#python train_model.py --embeddings output/PyPower_embed.pickle --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle
# importing nescessasry packages fromm library
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# here we constructed argument parser and thus parsed  the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="tells about path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="defines about the path to the output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="defines about the path to the output label encoder")
args = vars(ap.parse_args())

# here wehave loaded the face embeddings
print("[INFO] here we can see loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

#here we have encoded the labels
print("[INFO] here we can see encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])


#here we have tranied the model which help for accepting 128 d embedding of the face
# which is helpful in produciung the real face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

#  below method is used to writing face recognition model to the disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# here label encoder is written to disk 
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
