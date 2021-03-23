import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image
from PIL import Image
from skimage import transform
import io
import requests
import time
from googlesearch.googlesearch import GoogleSearch

app = Flask(__name__)

with open("data.pickle", "rb") as f:
	words, labels, training, output = pickle.load(f)
with open("intents.json", encoding="utf8") as file:
	data = json.load(file)
model = tf.keras.models.load_model('saved_model/my_model')
def pred(imgZ):
    test_image = image.load_img(b ,target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    cnn_result = cnn.predict(test_image)
    return cnn_result
def load(filename):
    np_image = filename
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (64, 64, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def canDetect(imag):
	cnn = tf.keras.models.load_model('TrainedModel_1')
	image = load(imag)
	result = cnn.predict(image)
	if result[0][0] >= .7:
	    return f'Image is Malignant'
	else:
	    return f'Image is Benign'

def bag_of_words(s, words):
	stemmer = LancasterStemmer()
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
	return np.array(bag)

@app.route("/")
def hello():
    return "Active bot"

past = []
ImgUrl = []
@app.route("/sms", methods=['POST'])
def sms_reply():
	try:
		file_count=request.form.get("NumMedia")
	except Exception as e:
		pass
	msg = request.form.get('Body')
	resp = MessagingResponse()
	past.append(int(file_count))
	temp = past[0]
	if (file_count == '1'):
		ImgUrl.append(str(request.form.get("MediaUrl0")))
		resp.message("Received Image\n⚠️ Should I forward this to Neural-Network-> (yes/no)")
	elif (temp == 1) and (str(msg).lower() in ['y', "yes",'n', "no"]):
		if str(msg).lower() in ['y', "yes"]:
			resp.message("Forwarding image to Neural-Network...")
			if ImgUrl[0] != "null":
				response = requests.get(ImgUrl[0])
				image_bytes = io.BytesIO(response.content)
				img = Image.open(image_bytes)
				output1 = canDetect(img)
				resp.message(output1)
				past.pop(0)
				ImgUrl.pop(0)
			else:
				resp.message("Forwarding denied")
		elif str(msg).lower() in ['n', "no"]:
			resp.message(f"Process Cancelled!\nReason user response: {msg}")
			past.pop(0)
			ImgUrl.pop(0)
		else:
			past.append(1)
			ImgUrl.append("null")
	else:
		past.clear()
		ImgUrl.clear()
		results = np.array([bag_of_words(str(msg).lower(), words)])
		results = model.predict(results)[0]
		result_index = np.argmax(results)
		tag = labels[result_index]
		if results[result_index] > 0.7:
			for tg in data["intents"]:
				if tg["tag"] == tag:
					responces = tg["responses"]
			op = random.choice(responces)

			if op in ["<DETECT>","<SAFEBM>"]:
				if op  == "<SAFEBM>":
					resp.message("Please Enter your pincode to find you nearby specialist.")
					
				elif op == "<DETECT>":
					resp.message("This query is removed from the chatbot")
			else:
				resp.message(op)
		else:
			if str(msg).isdigit() and len(str(msg))>4:
				pincode = msg
				response = GoogleSearch().search(f"cancer specialist near {str(pincode)}")
				for result1 in response.results:
				    resp.message(f"-> {result1.url}")
			else:
				resp.message(f"Sorry I did't get that, can you rephrase your query!")
	return str(resp)

if __name__ == "__main__":
	app.run(port=5000, debug=False)