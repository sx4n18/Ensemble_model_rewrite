import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize

########################################################################################################################
#
#   This is the script of a 3 layer ANN neural network for the simple 100 channel 8 classes isotopes dataset.
#
#
########################################################################################################################

## load the dataset
simple_radio_ds = pd.read_csv('./dataset/input.csv', header=None)
simple_radio_lb = pd.read_csv('./dataset/label.csv', header=None)
Train_y = to_categorical(simple_radio_lb-1)

Test_X = np.load('./dataset/x_test.npz')['arr_0']
Test_y = np.load('./dataset/y_test.npz')['arr_0']

## build the simple model
initializer = keras.initializers.HeNormal()

model = keras.Sequential([
        keras.layers.Dense(40, activation='relu', kernel_initializer=initializer),
        keras.layers.Dense(8, activation='softmax', kernel_initializer=initializer)
        ])

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(simple_radio_ds,Train_y,
                    epochs=200,
                    shuffle=True,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)]
                    )
#model.save('Simple_single_3_layer_NN.h5')

model.evaluate(Test_X,Test_y)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2)
plt.show()