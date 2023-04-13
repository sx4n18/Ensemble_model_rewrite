from tensorflow import keras
import numpy as np
import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import itertools
from fun_n_class.main_fun import major_hardvote
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

################################################################################################
# Since it will be way too big to fit all 20 networks inside the microcontroller Arduino nano,
# It would be good if we could simply just fit in 4-5 networks inside and achieve comparable
# accuracy, this script will dig inside the combination and find out the best performing combo
################################################################################################

diagonal_num = 20
total_sample_number = 9000
total_testing_number = int(0.15*total_sample_number)
total_cls_num = 18
each_prediction = np.zeros(shape=(diagonal_num, total_testing_number, total_cls_num))
label = np.load('../data_set/label.npy')
label_minus_1 = label - 1
label_1_hot = to_categorical(label_minus_1)
X_train_lst = []
X_test_lst = []
model_lst = []
max_acc = 0
max_acc_lst_5 = []
max_combo_5 = []
for dia_num in range(diagonal_num):
    model_this_dia = keras.models.load_model('../Ensemble_CP/HDF5/diagonal_'+str(dia_num)+'.h5')
    model_lst.append(model_this_dia)
    raw_np_file = np.load('../data_set/diagonal_' + str(dia_num) + '.npy')
    Train_X, Test_X, Train_y, Test_y = train_test_split(raw_np_file, label_1_hot, test_size=0.15,
                                                        random_state=42, shuffle=True)

    X_train_lst.append(Train_X)
    X_test_lst.append(Test_X)
    each_prediction[dia_num] = model_this_dia.predict(Test_X)
print("\n")
print("Now trying the combination of random chosen 5 networks and the accuracy")

actual_lb = np.argmax(Test_y, axis=1)

for combo in tqdm.tqdm(itertools.combinations(range(20), 5), ncols=100):
    extracted_prediction = each_prediction[list(combo)]
    hardvoting_lb_iter = major_hardvote(extracted_prediction)
    acc_this_time = accuracy_score(actual_lb, hardvoting_lb_iter)
    if acc_this_time > max_acc:
        max_acc = acc_this_time
        max_acc_lst_5.append(max_acc)
        max_combo_5.append(combo)

print("There are ", len(max_acc_lst_5), "different combos")
print(max_acc_lst_5)
print(max_combo_5)

max_acc = 0
max_acc_lst_7 = []
max_combo_7 = []
print("\n")
print("Now trying the combination of random chosen 7 networks and the accuracy")

for combo in tqdm.tqdm(itertools.combinations(range(20), 7), ncols=100):
    extracted_prediction = each_prediction[list(combo)]
    hardvoting_lb_iter = major_hardvote(extracted_prediction)
    acc_this_time = accuracy_score(actual_lb, hardvoting_lb_iter)
    if acc_this_time > max_acc:
        max_acc = acc_this_time
        max_acc_lst_7.append(max_acc)
        max_combo_7.append(combo)

print("There are ", len(max_acc_lst_7), "different combos")
print(max_acc_lst_7)
print(max_combo_7)

max_acc = 0
max_acc_lst_8 = []
max_combo_8 = []
print("\n")
print("Now trying the combination of random chosen 8 networks and the accuracy")

for combo in tqdm.tqdm(itertools.combinations(range(20), 8), ncols=100):
    extracted_prediction = each_prediction[list(combo)]
    hardvoting_lb_iter = major_hardvote(extracted_prediction)
    acc_this_time = accuracy_score(actual_lb, hardvoting_lb_iter)
    if acc_this_time > max_acc:
        max_acc = acc_this_time
        max_acc_lst_8.append(max_acc)
        max_combo_8.append(combo)

print("There are ", len(max_acc_lst_8), "different combos")
print(max_acc_lst_8)
print(max_combo_8)