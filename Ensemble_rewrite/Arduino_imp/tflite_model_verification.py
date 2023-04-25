import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from fun_n_class.main_fun import tflite_prediction_with_int8_input
from fun_n_class.main_fun import give_final_result_from_ensemble
from sklearn.metrics import accuracy_score

################################################################################################
# This is a simple script to verify the accuracy of the tflite model converted, at the same time
# it will explore the usage of the tf.lite interpreter for models that have integer input.
################################################################################################


chosen_combo_5 = np.array([5, 8, 11, 17, 19])
actual_lb = np.load('../data_set/label.npy')
label_1_hot = to_categorical(actual_lb-1)
X_train_lst = []
X_test_lst = []

# get the training and testing data ready for validation
for item in chosen_combo_5:
    raw_np_file = np.load('../data_set/log_preprocess/log_plus_1_subtraction_prepro/diagonal_preprocess_log_' + str(item) + '.npy')
    Train_X, Test_X, Train_y, Test_y = train_test_split(raw_np_file, label_1_hot, test_size=0.15,
                                                        random_state=42, shuffle=True)
    X_test_lst.append(Test_X)
    X_train_lst.append(Train_X)

X_train_concat = np.concatenate(X_train_lst, axis=1)
X_test_concat = np.concatenate(X_test_lst, axis=1)

test_actual_label = np.argmax(Test_y, axis=1)

# load the big ensemble tflite model

model_tflite = tf.lite.Interpreter('./tflite_model/whittled_ensemble_5.tflite')

model_verification = tflite_prediction_with_int8_input(model_tflite, X_test_concat.shape[0],
                                                       X_test_concat.shape[1], 90, X_test_concat)

final_result = give_final_result_from_ensemble(model_verification, 18)
acc = accuracy_score(test_actual_label, final_result)

print("Accuracy from tflite ensemble is: {acc:.2f} %".format(acc=acc*100))

print(acc)

