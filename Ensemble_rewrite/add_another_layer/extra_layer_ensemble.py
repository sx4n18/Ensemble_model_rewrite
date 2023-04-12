
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from fun_n_class.main_fun import major_hardvote
from fun_n_class.main_fun import sub_cls_to_sequential
import itertools
import matplotlib.pyplot as plt
import tqdm

##############################################################################
# This is a script to add an extra layer to the final ensemble decision-making
# at the same time, this will explore how many networks we need in the ensemble
# model to have a comparable accuracy.
##############################################################################



## verify accuracy before transfer learning

## load the dataset

diagonal_num = 20
raw_data_lst = []
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


for i in range(diagonal_num):
    raw_np_file = np.load('../data_set/diagonal_' + str(i) + '.npy')
    Train_X, Test_X, Train_y, Test_y = train_test_split(raw_np_file, label_1_hot, test_size=0.15,
                                                        random_state=42, shuffle=True)
    raw_data_lst.append(raw_np_file)
    X_train_lst.append(Train_X)
    X_test_lst.append(Test_X)
    try:
        model_this_dia = keras.models.load_model('../Ensemble_CP/HDF5/diagonal_'+str(i)+'.h5')
    except:
        sub_cls_model_this_dia = keras.models.load_model('../Ensemble_CP/diagonal_' + str(i))
        model_this_dia = sub_cls_to_sequential(sub_cls_model_this_dia, i)
        model_this_dia.save('../Ensemble_CP/HDF5/diagonal_' + str(i) + '.h5')
    model_lst.append(model_this_dia)
    each_prediction[i] = model_this_dia.predict(Test_X)

hardvoting_lb = major_hardvote(each_prediction)
actual_lb = np.argmax(Test_y, axis=1)
ensemble_acc = accuracy_score(actual_lb, hardvoting_lb)

print("\n")
print("Pretrained ANN models loaded and accuracy with the ensemble model is: {acc:.2%}".format(acc=ensemble_acc))

## now iterate all possible combinations of each choice of number of networks in the ensemble model and get the overall accuracy and plot out

print("Now trying to get the relation between number of networks and overall accuracy")

################################################################################################################
# Since the following part will take a really long time to run, most of the time, this should be commented off.
################################################################################################################
# lst_of_acc = []

## add tqdm to see where exactly we are
'''
for num_of_iter in tqdm.tqdm(range(1,21,1)):
    acc_lst_this_number_of_iter = []
    for iterable in itertools.combinations(range(20), num_of_iter):
        extracted_prediction = each_prediction[list(iterable)]
        hardvoting_lb_iter = major_hardvote(extracted_prediction)
        acc_this_time = accuracy_score(actual_lb, hardvoting_lb_iter)
        acc_lst_this_number_of_iter.append(acc_this_time)
    lst_of_acc.append(acc_lst_this_number_of_iter)

plt.boxplot(lst_of_acc)
plt.show()
'''
################################################################################################################
# Since the above part will take a really long time to run, most of the time, this should be commented off.
################################################################################################################


# build the new network with pretrained network from previous work


layer_concat = [model.output for model in model_lst] ## concatenate all the output
input_layer_concat = [model.input for model in model_lst] ## concatenate all the input
concat_lay = keras.layers.concatenate(layer_concat, name='Concated_layers') ## make the concatenate layer
drop_out_lay = keras.layers.Dropout(0.5)(concat_lay)
ensemble_output_layer = keras.layers.Dense(18, activation='softmax', name='Ensemble_output', kernel_regularizer='l1', use_bias=False)(drop_out_lay) ## pass the concatenate layer into the output layer
transfer_learning_model = keras.Model(inputs=input_layer_concat, outputs=ensemble_output_layer)

print(transfer_learning_model.input_shape)
print(transfer_learning_model.output_shape)

transfer_learning_model.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
transfer_learning_model.trainable = False ## freeze all the parameters
transfer_learning_model.layers[-1].trainable = True ## turn on last layers training ability

history = transfer_learning_model.fit(X_train_lst, Train_y,
                                      epochs=10,
                                      shuffle=True,
                                      validation_split=0.18)

transfer_learning_model.evaluate(X_test_lst, Test_y)


weights_last_lay = transfer_learning_model.variables
print(weights_last_lay)


