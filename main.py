#%pip install numpy
#%pip install pandas
#%pip install matplotlib
#%pip install nltk
#%pip install tensorflow

import random
import pickle
import heapq

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

import numpy as np
from numpy import savetxt
from numpy import save
from numpy import load
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

import os



n_words = 10

cwd = os.getcwd()
absolute_path = os.path.realpath(__file__)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#print('New working directory is: ', os.getcwd())




def read_text(texto): #lee el texto con el cual se desea entrenar la red neuronal, en este caso es un archivo csv
    #Esta parte lee un archivo csv, saca solo los texto del mismo y luegos los junta en un archivo txt.·
    text_df = pd.read_csv(texto,engine ='python')
    text = list(text_df.text.values)
    joined_text = " ".join(text)
    print ("-........------------------")
    with open("data/joined_text.txt", "w", encoding="utf-8") as f:
        f.write(joined_text)
    partial_text = joined_text[:40000]
    

def Train_new_model():
    joined_text = open("data/text.txt", "r", encoding="latin-1").read()

    partial_text = joined_text[:250000]

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(partial_text.lower())

    unique_tokens = np.unique(tokens)
    unique_token_index = {token: index for index, token in enumerate(unique_tokens)}
    n_words = 10  #El modelo funciona leyendo n Palabras de un texto y de alli saca la mas posible siguiente palabra
    input_words = []
    next_word = []

    for i in range(len(tokens) - n_words):
        input_words.append(tokens[i:i + n_words])
        next_word.append(tokens[i + n_words])

    X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)  # para cada muestra, se da n Palabras entrante luego un valor booleano para cada posible siguiente palabra
    y = np.zeros((len(next_word), len(unique_tokens)), dtype=bool)  # Por cada muestra, un booleano por cada posible siguiente palabra



    for i, words in enumerate(input_words):
        for j, word in enumerate(words):
            X[i, j, unique_token_index[word]] = 1
        y[i, unique_token_index[next_word[i]]] = 1

    model = Sequential()
    Layer1= model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
    Layer2 = model.add(LSTM(128))
    Layer3 = model.add(Dense(len(unique_tokens)))
    Layer4 = model.add(Activation("softmax"))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    history = model.fit(X, y, batch_size=128, epochs=10, shuffle=True).history
    model.save("text_gen_model2.h5")

    save ('data/unique_tokens', unique_tokens)
    save ('data/unique_token_index', unique_token_index)

def Train_current_model (model_name, text): #Funcion para seguir entrenando el modelo com mas textos, hasta ahora no esta completo

    partial_text = text

    unique_tokens =load('unique_tokens.npy')
    unique_token_index = load('unique_token_index.npy')
    model = load_model("text_gen_model2.h5")

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(partial_text.lower())

    n_words = 10
    input_words = []
    next_word = []

    for i in range(len(tokens) - n_words):
        input_words.append(tokens[i:i + n_words])
        next_word.append(tokens[i + n_words])
    X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)  
    y = np.zeros((len(next_word), len(unique_tokens)), dtype=bool)  

    history = model.fit(X, y, batch_size=128, epochs=10, shuffle=True).history
    model.save("text_gen_model2.h5")


def predict_next_word(input_text, n_best):

    unique_tokens =load('data/unique_tokens.npy')
    unique_token_index = load('data/unique_token_index.npy')
    model = load_model("text_gen_model2.h5")

    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1
        
    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

def generate_text(input_text, n_words, creativity=3): #Genera un texto
    unique_tokens =load('data/unique_tokens.npy')
    tokenizer = RegexpTokenizer(r"\w+")
    word_sequence = input_text.split()
    current = 0
    for _ in range(n_words):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)
   

#read_text()
#Train_new_model()

term = input("Escribe 1 para entrenar el modelo, 2 para generar un texto de ejemplo: ")

match term:
    case "1":
        Train_new_model()
    case "2":
        term1 = input("Escribe el numero de palabras que desee generar ")
        term2 = input("Escribe la creatividad del generador ")
        term1 = int(term1)  # parse string into an integer
        term2 = int(term2)  # parse string into an integer
        print (generate_text("Jose ha anunciado que", term1, term2))
    case _:
        print ("not valid")

