import tensorflow.keras as keras
#import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from fun_n_class.main_cls import data_set_generator
from fun_n_class.main_fun import give_final_result_from_ensemble
from sklearn.metrics import accuracy_score

################################################################################################
# This script will try to concatenate the chosen 5 networks and see how much they would take in
# the form of tflite.
################################################################################################

chosen_combo_5 = np.array([5, 8, 11, 17, 19])
model_lst = []
concat_input = []
concat_output = []
actual_lb = np.load('../data_set/label.npy')
label_1_hot = to_categorical(actual_lb-1)
X_train_lst = []
X_test_lst = []

# reconstruct the ensemble net with just the chosen 5 networks

for item in chosen_combo_5:
    diagonal_net = keras.models.load_model('../Ensemble_CP_log_plus_1_w_subtraction/HDF5/diagonal_'+str(item)+'.h5')
    raw_np_file = np.load('../data_set/log_preprocess/log_plus_1_subtraction_prepro/diagonal_preprocess_log_'+str(item)+'.npy')
    Train_X, Test_X, Train_y, Test_y = train_test_split(raw_np_file, label_1_hot, test_size=0.15,
                                                        random_state=42, shuffle=True)
    X_test_lst.append(Test_X)
    X_train_lst.append(Train_X)
    model_lst.append(diagonal_net)


#concated_input_lay = keras.layers.concatenate(concat_input, name='concat_input_layer')
#concated_output_lay = keras.layers.concatenate(concat_output, name='concat_output_layer')

## create a uni input and break it down into different range for the different net's feed
input_tensor = keras.layers.Input(5055,)
group_1 = keras.layers.Lambda(lambda x: x[:, :1018], output_shape=(1023,))(input_tensor)
group_2 = keras.layers.Lambda(lambda x: x[:, 1018:2033], output_shape=(1021,))(input_tensor)
group_3 = keras.layers.Lambda(lambda x: x[:, 2033:3045], output_shape=(1013,))(input_tensor)
group_4 = keras.layers.Lambda(lambda x: x[:, 3045:4051], output_shape=(1007,))(input_tensor)
group_5 = keras.layers.Lambda(lambda x: x[:, 4051:], output_shape=(1006,))(input_tensor)

## feed different net with different part of the input and get the output tensor
x_out_1 = model_lst[0](group_1)
x_out_2 = model_lst[1](group_2)
x_out_3 = model_lst[2](group_3)
x_out_4 = model_lst[3](group_4)
x_out_5 = model_lst[4](group_5)

## concatenate the output tensor from the sub-module
concat_output_lay = keras.layers.Concatenate()([x_out_1, x_out_2, x_out_3, x_out_4, x_out_5])

## make the whole model stand from end to end
new_ensemble_net = keras.Model(inputs=input_tensor, outputs=concat_output_lay)

print(new_ensemble_net.input_shape)
print(new_ensemble_net.output_shape)

## save this model for later observation
new_ensemble_net.save('./whittled_ensemble_model/whittled_net_5.h5')
new_ensemble_net.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                         metrics=['accuracy'])
print("Verify the accuracy of the concated net")
Test_label = [Test_y] * 5
concated_Test_img = np.concatenate(X_test_lst, axis=1)
concated_Test_label = np.concatenate(Test_label, axis=1)
ensemble_prediction = new_ensemble_net.predict(concated_Test_img)
voted_prediction = give_final_result_from_ensemble(ensemble_prediction, 18)
acc = accuracy_score(np.argmax(Test_y, axis=1), voted_prediction)
#_, acc = new_ensemble_net.evaluate(concated_Test_img, concated_Test_label)
print("Accuracy is :", acc)

# convert the network to tflite and quantisation

## get representative dataset ready
concated_Train_img = np.concatenate(X_train_lst, axis=1)
concated_Train_img = concated_Train_img.astype(np.float32)
generator = data_set_generator(concated_Train_img)

## configure the quantization settings and convert
converter = tf.lite.TFLiteConverter.from_keras_model(new_ensemble_net)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
quantised_model = converter.convert()

## save the quantised model
quantised_model_tflite = open('./tflite_model/whittled_ensemble_5.tflite', 'wb').write(quantised_model)

