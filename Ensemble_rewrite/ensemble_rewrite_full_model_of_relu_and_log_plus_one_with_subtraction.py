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
#from tensorflow.keras.utils import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

########################################################################################################################
#                                                                                                                      #
#   This is the script of an ensemble model of 20 3-layer ANN neural networks for the 1024 channel 18 classes
#   isotopes dataset.                                                                                                  #
#   This is the 3rd version where activation function has been replaced with relu and input raw dataset was first
#   logarithmically preprocessed and then handled with subtraction
#   so that the conversion of this model to SNN would be more convenient.
########################################################################################################################

## class functin define

## class for each small ANN implementation
class ANN_simple_model(keras.Model):

    def __init__(self, input_num, initializer):
        super(ANN_simple_model, self).__init__()
        self.input_num = input_num
        self.initializer = initializer
        self.hidden = keras.layers.Dense(40, activation='relu', kernel_initializer=self.initializer)
        self.outlayer = keras.layers.Dense(18, activation='softmax', kernel_initializer=self.initializer)
        self.dplay1 = keras.layers.Dropout(rate=0.25)
        self.dplay2 = keras.layers.Dropout(rate=0.25)

    def call(self, input_tensor, training=True, **kwargs):
        x = self.dplay1(input_tensor)
        # layer construction
        x = self.hidden(x)

        x = self.dplay2(x)

        return self.outlayer(x)

## Function of the ensemble hard/soft voting
## input: a 3 dimensional array that records the raw output of the network (probability), (diagonal_number, sample_number, class_number)
## output: just a majority vote of the final classification
def major_hardvote(all_diagonal_prediction):
    final_class_of_each_ANN = np.argmax(all_diagonal_prediction, axis=2)
    final_prediction_after_voting = np.zeros((total_testing_number),'int')
    for sample_index in range(total_testing_number):
        bin_count = np.bincount(final_class_of_each_ANN[:, sample_index])
        final_prediction_after_voting[sample_index] = np.argmax(bin_count)

    return final_prediction_after_voting

def major_softvote(all_diagona_prediction):
    overall_odds_from_each_ANN = np.sum(all_diagona_prediction, axis=0)
    final_prediction_after_voting = np.argmax(overall_odds_from_each_ANN,axis=1)

    return final_prediction_after_voting


## load the dataset

diagonal_num = 20
raw_data_lst = []
total_sample_number = 9000
total_testing_number = int(0.15*total_sample_number)
total_cls_num = 18
each_prediction = np.zeros(shape=(diagonal_num,total_testing_number,total_cls_num))
label = np.load('./data_set/label.npy')
label_minus_1 = label - 1
label_1_hot = to_categorical(label_minus_1)

for i in range(diagonal_num):
    raw_np_file = np.load('./data_set/log_preprocess/log_plus_1_subtraction_prepro/diagonal_preprocess_log_' + str(i) + '.npy')
    #raw_np_file = np.nan_to_num(raw_np_file, nan=0, neginf=0, posinf=0)
    raw_data_lst.append(raw_np_file)

## build model and training

initializer = keras.initializers.HeNormal()

for index, single_diagonal in enumerate(raw_data_lst):

    path = './Ensemble_CP_log_plus_1_w_subtraction/diagonal_' + str(index)
    if not (os.path.isdir(path)):
        os.makedirs(path)
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

#np.save('./prediction/final_raw_prediction.npy', each_prediction)

## final accuracy by majority vote and print out and produce a confusin matrix
final_predicted_lb_hard_voting = major_hardvote(each_prediction)
final_predicted_lb_soft_voting = major_softvote(each_prediction)
actual_lb = np.argmax(Test_y, axis=1)
cf_matrix = confusion_matrix(actual_lb, final_predicted_lb_hard_voting)
#diff = actual_lb - final_predicted_lb_hard_voting
#wrong = np.count_nonzero(diff)
#final_accuracy = (total_testing_number-wrong)/total_testing_number
final_accuracy_hard = accuracy_score(actual_lb, final_predicted_lb_hard_voting)
final_accuracy_soft = accuracy_score(actual_lb, final_predicted_lb_soft_voting)
print('Final prediction based on soft voting gives a test accuracy of {acc:.2%}'.format(acc=final_accuracy_soft))
print('Final prediction based on hard voting gives a test accuracy of {acc:.2%}'.format(acc=final_accuracy_hard))

## plot out the confusion matrix
ax = sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')

ax.set_title('Confusion matrix of the Ensemble learning model')
ax.set_ylabel('Actual label')
ax.set_xlabel('\nPredicted label')

## Ticket labels - List must be in alphabetical order
All_the_label = ['Am241', 'Ba133', 'BGD', 'Co57', 'Co60', 'Cs137', 'DU', 'EU152', 'Ga67', 'HEU', 'I131', 'Ir192', 'Np237', 'Ra226', 'Tc99m', 'Th232', 'Tl201', 'WGPu']
ax.xaxis.set_ticklabels(All_the_label)
ax.yaxis.set_ticklabels(All_the_label)

## Display the visualization of the Confusion Matrix.
plt.show()