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

## performance_oriented callback definition
K = keras.backend

class performance_goal_callback(keras.callbacks.Callback):

    def __init__(self, patience=5):
        super(performance_goal_callback, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.goal = 0.005

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        if np.greater(current, self.goal):
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)


## function definition

def get_dataset_partitions_tf(ds, ds_size, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


## load the dataset for just one diagonal

X_full = np.load('./data_set/diagonal_0.npy')
X_full = X_full.reshape((9000, 1, 1023, 1))
label = np.load('./data_set/label.npy')
label_minus_1 = (label - 1)  # .reshape(9000, 1)
label_one_hot = to_categorical(label_minus_1)
label_one_hot = label_one_hot.reshape((9000, 1, 18))
full_ds = tf.data.Dataset.from_tensor_slices((X_full, label_one_hot))

Train_ds, Val_ds, Test_ds = get_dataset_partitions_tf(full_ds, 9000, 0.7, 0.15, 0.15)

## extract the numpy data from dataset type

Train_X = np.asarray(list(Train_ds.map(lambda x, y: x)))
Train_y = np.asarray(list(Train_ds.map(lambda x, y: y)))
a,b,c,d = Train_X.shape
Train_X = Train_X.reshape((a,c,d))
Train_X = normalize(Train_X)
a,b,c = Train_y.shape
Train_y = Train_y.reshape(a,c)
Valid_X = np.asarray(list(Test_ds.map(lambda x, y: x)))
Valid_y = np.asarray(list(Test_ds.map(lambda x, y: y)))
a,b,c,d = Valid_X.shape
Valid_X = Valid_X.reshape((a,c,d))
Valid_X = normalize(Valid_X)
a,b,c = Valid_y.shape
Valid_y = Valid_y.reshape(a,c)
## model defination


initializer = keras.initializers.HeNormal()
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[1023, 1]),
    keras.layers.Dense(40, activation='sigmoid', kernel_initializer=initializer,  kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(18, activation='softmax', kernel_initializer=initializer, kernel_regularizer=keras.regularizers.l2(0.01))
])

model.summary()
keras.utils.plot_model(model, "my_model.png", show_shapes=True)

model.compile(#loss='mse',
              loss=categorical_crossentropy,
              # optimizer= keras.optimizers.SGD(learning_rate = 0.001),
              #optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999),
              #optimizer="adam",
              metrics=['accuracy'])

expon_lr = ExponentialLearningRate(factor=1.005)

history = model.fit(Train_X,Train_y,
          epochs=1000,
          steps_per_epoch=1260,
          # steps_per_epoch=630,
          validation_data=(Valid_X,Valid_y),
          #callbacks=[expon_lr]
          callbacks=[keras.callbacks.EarlyStopping(patience=10)]
          )

#model.save('my_model_single_nn.h5')
#model.evaluate(Test_ds)
#Test_X = np.asarray(list(Test_ds.map(lambda x, y: x)))
#Test_y = np.asarray(list(Test_ds.map(lambda x, y: y)))
#print(model.predict(Test_X.reshape(1350, 1023, 1)))

#plt.plot(expon_lr.rates, expon_lr.losses)
#plt.gca().set_xscale('log')
#plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
#plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
#plt.grid()
#plt.xlabel("Learning rate")
#plt.ylabel("Loss")
#plt.show()

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 4)
plt.show()