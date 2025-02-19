import tensorflow as tf
import pandas as pd 
import numpy as np
import seaborn as sns
import scipy.io as scio

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from sys import path 
path.append("../../utils")
from dataset_operations import *
from utils import *
from models import build_model 
from load_federated_data import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATASET = "CIFAR"    

MODEL = "EFFICIENTNET"
#MODEL = "MOBILENET"


trPer=0.8
x_train, y_train, x_test, y_test = load_data_federated_IID("CIFAR-10", 
                                                            1, 
                                                            1, 
                                                            trPer)


# Defining the deep learning model
if MODEL == "MOBILENET":
    model = tf.keras.applications.MobileNet((32, 32, 3), 
                                             classes=10,
                                             weights=None)
elif MODEL == "VGG19":
    model = tf.keras.applications.VGG19(input_shape=(32, 32, 3), 
                                        classes=10,
                                        weights=None)

elif MODEL == "EFFICIENTNET":
    model = tf.keras.applications.EfficientNetV2L(input_shape=(32, 32, 3), 
                                                  classes=10,
                                                  weights=None)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                         patience=3)


model.fit(x_train,y_train,epochs=50,batch_size=32, callbacks=[callback])
    
model.evaluate(x_test,y_test)
    
model.save("../../models/centralized_"+DATASET+".keras")





