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
#                                                                                                                      #
#   This is the script of an ensemble model of 20 3-layer ANN neural networks for the 1024 channel 18 classes
#   isotopes dataset.                                                                                                  #
#
#
########################################################################################################################

## class functin define
class ANN_simple_model(keras.Model):

    def __init__(self, input_num, initializer):
        super(ANN_simple_model, self).__init__()
        self.input_num = input_num
        self.initializer = initializer
        self.hidden = keras.layers.Dense(40, activation='sigmoid', kernel_initializer=self.initializer)
        self.outlayer = keras.layers.Dense(18, activation='softmax', kernel_initializer=self.initializer)
        self.dplay1 = keras.layers.Dropout(rate=0.25)
        self.dplay2 = keras.layers.Dropout(rate=0.25)

    def call(self, input_tensor, training=True, **kwargs):
        x = self.dplay1(input_tensor)
        # layer construction
        x = self.hidden(x)

        x = self.dplay2(x)

        return self.outlayer(x)


## load the dataset

diagonal_num = 20
raw_data_lst = []
each_prediction = np.zeros(shape=(diagonal_num,1350,18))
label = np.load('./data_set/label.npy')
label_minus_1 = label - 1
label_1_hot = to_categorical(label_minus_1)

for i in range(diagonal_num):
    raw_np_file = np.load('./data_set/diagonal_' + str(i) + '.npy')
    raw_data_lst.append(raw_np_file)

## build model and training

initializer = keras.initializers.HeNormal()

for index, single_diagonal in enumerate(raw_data_lst):

    path = './Ensemble_CP/diagonal_' + str(index)
    if not (os.path.isdir(path)):
        os.mkdir(path)
    #hdf5_path = os.path.join(path, 'best_model.h5')
    Train_X, Test_X, Train_y, Test_y = train_test_split(single_diagonal, label_1_hot, test_size=0.15,
                                                        random_state=42, shuffle=True)
    sample_num, input_num = Test_X.shape
    small_ANN_model = ANN_simple_model(input_num=input_num, initializer=initializer)
    small_ANN_model.compile(loss=categorical_crossentropy,
                            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                            metrics=['accuracy'])

    save_best_val_loss_cb = keras.callbacks.ModelCheckpoint(filepath=path,
                                                            monitor='val_loss',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            mode='min'
                                                            )
    small_ANN_model.fit(Train_X, Train_y,
                        epochs=1000,
                        batch_size=100,
                        shuffle=True,
                        validation_split=0.18,
                        callbacks=[keras.callbacks.EarlyStopping(patience=50), save_best_val_loss_cb]
                        )

    each_prediction[index] = small_ANN_model.predict(Test_X)

np.save('./prediction/final_raw_prediction.npy', each_prediction)

