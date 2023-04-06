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
from callbacks_and_cls import ExponentialLearningRate

########
#
#   This is the script of the ANN model for the 1st diagonal data to test the best perfoming dropout rate
#
#
#######

## class and function defination

class single_ANN_imp_with_parameter(keras.Model):

    def __init__(self, initializer, dropout_rate):
        super(single_ANN_imp_with_parameter, self).__init__()
        self.initializer = initializer
        self.dp_rate = dropout_rate
        self.hidden = keras.layers.Dense(40, activation='relu', kernel_initializer=self.initializer)
        self.outlayer = keras.layers.Dense(18, activation='softmax', kernel_initializer=self.initializer)
        self.dplay1 = keras.layers.Dropout(rate=self.dp_rate)
        self.dplay2 = keras.layers.Dropout(rate=self.dp_rate)

    def call(self, input_tensor, training=True, **kwargs):
        x = self.dplay1(input_tensor)
        # layer construction
        x = self.hidden(x)

        x = self.dplay2(x)

        return self.outlayer(x)

## load the dataset and preprocessing

whole_sample = np.load('./data_set/diagonal_0.npy')
label = np.load('./data_set/label.npy')
label_minus_1 = label-1
Train_X, Test_X, Train_y, Test_y = train_test_split(whole_sample, label_minus_1, test_size=0.15,
                                                    random_state=42, shuffle=True)

y_Train_cat = to_categorical(Train_y)
y_Test_cat = to_categorical(Test_y)

dropout_rate_lst = [item*0.02 for item in range(41)]
models_lst = []
score_1 = []
score_2 = []
## build up the model
initalizer = keras.initializers.HeNormal()

for rate in dropout_rate_lst:

    model_w_diff_config = single_ANN_imp_with_parameter(initializer=initalizer, dropout_rate=rate)

    model_w_diff_config.compile(loss=categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    model_w_diff_config.fit(Train_X,y_Train_cat,
                            epochs=1000,
                            batch_size=100,
                            shuffle=True,
                            validation_split=0.18,
                            callbacks=[keras.callbacks.EarlyStopping(patience=50)]
                            )

    models_lst.append(model_w_diff_config)

    score_w_each_config = model_w_diff_config.evaluate(Test_X, y_Test_cat)

    score_1.append(score_w_each_config[0])

    score_2.append(score_w_each_config[1])


plt.figure(1)
plt.plot(dropout_rate_lst, score_1, label='Test loss')
plt.plot(dropout_rate_lst, score_2, label='Test accuracy')
plt.title('Test loss and accuracy for each dropout rate')
plt.legend()
plt.show()

print('Best dropout rate found from range [0:0.8] is', dropout_rate_lst[score_2.index(max(score_2))])
print('The highest ccuracy for that is {acc:.2%}'.format(acc=max(score_2)))




