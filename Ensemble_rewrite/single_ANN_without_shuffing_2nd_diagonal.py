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
#   This is the script of the ANN model for the 1st diagonal data
#
#
#######
## load the dataset and preprocessing

whole_sample = np.load('./data_set/diagonal_1.npy')
label = np.load('./data_set/label.npy')
label_minus_1 = label-1
Train_X, Test_X, Train_y, Test_y =  train_test_split(whole_sample,label_minus_1,test_size=0.15,
                                                    random_state=42,shuffle=True)

y_Train_cat = to_categorical(Train_y)
y_Test_cat = to_categorical(Test_y)


## build up the model
initalizer = keras.initializers.HeNormal()

model = keras.Sequential([
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(40, activation='sigmoid', kernel_initializer=initalizer), #kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(18, activation='softmax', kernel_initializer=initalizer)#, kernel_regularizer=keras.regularizers.l2(0.001))
])
model.build(input_shape=(None,1022))
model.summary()
model.compile(loss=categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
              #optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              metrics=['accuracy'])

file_path = './CheckPointPath/my_best_model_single_nn.h5'
save_best_val_loss_cb = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                        monitor='val_loss',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        mode='min'
                                                        )

history = model.fit(Train_X, y_Train_cat,
                    epochs=1000,
                    batch_size=100,
                    shuffle=True,
                    validation_split=0.18,
                    callbacks=[keras.callbacks.EarlyStopping(patience=50),save_best_val_loss_cb]
                    )



#keras.utils.plot_model(model, "my_model_wo_flatten.png", show_shapes=True)

#model.save('single_ANN_wo_shuffling.h5')

model.evaluate(Test_X, y_Test_cat)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 4)
plt.show()
