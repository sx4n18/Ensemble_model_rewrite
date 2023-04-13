from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from fun_n_class.main_fun import major_hardvote

################################################################################################
# This script aims to get a rough idea of how good the best combos of nets are. This will simply
# give a verification.
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
best_combo_5 = np.array([0, 2, 10, 16, 17])
individual_acc = []
all_acc = []
for dia_num in range(diagonal_num):
    model_this_dia = keras.models.load_model('../Ensemble_CP/HDF5/diagonal_'+str(dia_num)+'.h5')
    model_lst.append(model_this_dia)
    raw_np_file = np.load('../data_set/diagonal_' + str(dia_num) + '.npy')
    Train_X, Test_X, Train_y, Test_y = train_test_split(raw_np_file, label_1_hot, test_size=0.15,
                                                        random_state=42, shuffle=True)

    X_train_lst.append(Train_X)
    X_test_lst.append(Test_X)
    each_prediction[dia_num] = model_this_dia.predict(Test_X)
    model_this_dia.evaluate(Test_X, Test_y)
print("\n")
print("Now trying the combination of random chosen 5 networks and the accuracy")

actual_lb = np.argmax(Test_y, axis=1)

for item in best_combo_5:
    predicted_lb = np.argmax(each_prediction[item], axis=1)
    acc = accuracy_score(actual_lb, predicted_lb)
    individual_acc.append(acc)

print(individual_acc)

extracted_prediction = each_prediction[best_combo_5]
predicted_lb_ensemble = major_hardvote(extracted_prediction)
overall_acc = accuracy_score(actual_lb, predicted_lb_ensemble)
print(overall_acc)
