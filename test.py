import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
stemmer = LancasterStemmer()
model = tf.keras.models.load_model('saved_model/my_model')
with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

with open("intents.json", encoding="utf8") as file:
    data = json.load(file)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chat():
    print("start talking with the bot (quit to exit)")
    while(1):
        inp = input("You: ").lower()
        results = np.array([bag_of_words(inp, words)])
        results = model.predict(results)[0]
        result_index = np.argmax(results) #returns index of the greatest value in our list
        tag = labels[result_index]
        if inp in  ['bye','break', 'exit', 'quit', 'bye', 'close']:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responces = tg["responses"]
            print("Bot: ",random.choice(responces))
            break
        elif results[result_index] > 0.70: #W setting the p-value 
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responces = tg["responses"]
            print("Bot: ",random.choice(responces))
        else:
            print("Bot: Sorry I did't get that, As I'm limited to the purpose!")
if __name__ == "__main__":
	chat()