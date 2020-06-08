from flask import Flask, render_template, request
from imageio import imsave, imread
from tensorflow.keras.models import model_from_json
from skimage.transform import resize
import numpy as np 
import re 
import sys 
import os 
import tensorflow.keras.models 
from PIL import Image
import tensorflow as tf
import base64
from tensorflow.keras.models import load_model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# def init(): 
# 	json_file = open('model.json','r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	loaded_model = model_from_json(loaded_model_json)
# 	#load woeights into new model
# 	loaded_model.load_weights("model.h5")
# 	print("Loaded Model from disk")

# 	#compile and evaluate loaded model
# 	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 	#loss,accuracy = model.evaluate(X_test,y_test)
# 	#print('loss:', loss)
# 	#print('accuracy:', accuracy)
# 	graph = tf.get_default_graph()
# 	# graph = tf.compat.v1.get_default_graph()

	# return loaded_model,graph


# global model, graph
# #initialize these variables
# # model, graph = init()
def get_model():	
	global model
	model = load_model("CNN_Model.h5")
	print("Model Loaded")
# global model
# model = load_model("CNN_Model.h5")
# graph = tf.get_default_graph()
app = Flask(__name__)

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(rb'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		# output.write(imgstr.decode('base64'))
		output.write(base64.b64decode(imgstr))
	


get_model()


@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	#read the image into memory
	x = imread('output.png',pilmode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	# #make it the right size
	x = resize(x,(28,28))
	# #convert to a 4D tensor to feed into our model
	# x = x.reshape(1,28,28,1)
	x = x.reshape(1,28,28,1)
	
	# print "debug2"
	#in our computation graph
	# with graph.as_default():
		#perform the prediction
		# out = model.predict(x)
		# print(out)
		# print(np.argmax(out,axis=1))
		# #convert the response to a string
		# response = np.array_str(np.argmax(out,axis=1))
		# return response			
	# model = load_model("CNN_Model.h5")	
	# global model 
	out = model.predict(x)
	print(out)
	print(np.argmax(out,axis=1))
	#convert the response to a string
	response = np.array_str(np.argmax(out,axis=1))
	return response	

if __name__ == "__main__":
	#decide what port to run the app in
	# port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	# app.run(host='0.0.0.0', port=port)
	app.run(debug=True)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
