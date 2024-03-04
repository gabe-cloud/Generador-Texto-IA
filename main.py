#%pip install numpy
#%pip install pandas
#%pip install matplotlib
#%pip install nltk
#%pip install tensorflow
#%pip install customtkinter

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
from tkinter import *
import customtkinter as ctk
import sys
from PIL import ImageTk, Image

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

import os

ctk.set_appearance_mode("dark")

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

def Train_current_model (model_name, text): #Funcion para seguir entrenando el modelo com más textos.

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
menuOne = ["100", "200", "300", "400", "500"]
menuTwo = ["1", "2", "3", "4", "5"]
# Create the main window
root = ctk.CTk()
root.geometry('400x500')
root.title("Generador de Texto")
root.resizable(False, False)
root.iconbitmap("logo.ico")

def show_popup():
    popup = ctk.CTkToplevel(root)
    popup.geometry("400x500")
    popup.resizable(False, False)
    popup.title("Generador de Texto")
    popup.iconbitmap("logo.ico")

    labelOne = ctk.CTkLabel(popup, text="Ingrese el número de palabras que desea generar:")
    labelOne.pack(padx=30, pady=30)
    optionOne = ctk.CTkComboBox(popup, values=menuOne)
    optionOne.pack(padx=30, pady=10)

    labelTwo = ctk.CTkLabel(popup, text="Ingrese nivel de creatividad que desea:")
    labelTwo.pack(padx=30, pady=30)
    optionTwo = ctk.CTkComboBox(popup, values=menuTwo)
    optionTwo.pack(padx=30, pady=10)

    optionOne.get() 
    optionTwo.get() 

    generarTexto = ctk.CTkButton(popup, text="Generar Texto", command=lambda: generacion_texto(optionOne.get(), optionTwo.get(), popup))
    generarTexto.pack(padx=30, pady=30)

    popup.grab_set()
    popup.wait_window()
    popup.grab_release()


def generacion_texto(term1, term2, toplevel):
    new_popup = ctk.CTkToplevel(root)
    new_popup.geometry("400x500")
    new_popup.resizable(False, False)
    new_popup.title("Generador de Texto")
    new_popup.iconbitmap("logo.ico")

    scroll_frame = ctk.CTkScrollableFrame(new_popup, orientation="horizontal")
    scroll_frame.pack(padx=30, pady=30, ipadx=50, ipady=50)


    text_label = ctk.CTkLabel(scroll_frame, text="Texto generado:")
    text_label.pack(padx=10, pady=10)

    term1_int = int(term1)
    term2_int = int(term2)
    texto_generado = generate_text("Se anuncia que", term1_int, term2_int)
    text_label.configure(text=texto_generado)

    toplevel.destroy()

    new_popup.grab_set()
    new_popup.wait_window()
    new_popup.grab_release()

def salir():
    sys.exit()


# Create the button frame

button_frame = ctk.CTkFrame(master=root)
button_frame.pack(fill="both", expand=True)

# Create the buttons and pack them within the frame
firstButton = ctk.CTkButton(master=button_frame, text="Entrenar Modelo", command=Train_new_model)
firstButton.pack(padx=30, pady=50)

secondButton = ctk.CTkButton(master=button_frame, text="Generar texto", command=show_popup)
secondButton.pack(padx=30, pady=30)

ThirdButton = ctk.CTkButton(master=button_frame, text="Salir", command=salir)
ThirdButton.pack(padx=30, pady=30)

# Run the main loop
root.mainloop()
