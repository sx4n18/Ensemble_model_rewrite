import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from fun_n_class.main_cls import data_set_generator

################################################################################################
# This script will try to concatenate the chosen 5 networks and see how much they would take in
# the form of tflite.
################################################################################################

chosen_combo_5 = np.array([0, 2, 10, 16, 17])
model_lst = []
concat_input = []
concat_output = []
actual_lb = np.load('../data_set/label.npy')
label_1_hot = to_categorical(actual_lb-1)
X_train_lst = []
X_test_lst = []

# reconstruct the ensemble net with just the chosen 5 networks

for item in chosen_combo_5:
    diagonal_net = keras.models.load_model('../Ensemble_CP/HDF5/diagonal_'+str(item)+'.h5')
    raw_np_file = np.load('../data_set/diagonal_'+str(item)+'.npy')
    Train_X, Test_X, Train_y, Test_y = train_test_split(raw_np_file, label_1_hot, test_size=0.15,
                                                        random_state=42, shuffle=True)
    X_test_lst.append(Test_X)
    X_train_lst.append(Train_X)
    model_lst.append(diagonal_net)


#concated_input_lay = keras.layers.concatenate(concat_input, name='concat_input_layer')
#concated_output_lay = keras.layers.concatenate(concat_output, name='concat_output_layer')

## create a uni input and break it down into different range for the different net's feed
input_tensor = keras.layers.Input(5070,)
group_1 = keras.layers.Lambda(lambda x: x[:, :1023], output_shape=(1023,))(input_tensor)
group_2 = keras.layers.Lambda(lambda x: x[:, 1023:2044], output_shape=(1021,))(input_tensor)
group_3 = keras.layers.Lambda(lambda x: x[:, 2044:3057], output_shape=(1013,))(input_tensor)
group_4 = keras.layers.Lambda(lambda x: x[:, 3057:4064], output_shape=(1007,))(input_tensor)
group_5 = keras.layers.Lambda(lambda x: x[:, 4064:], output_shape=(1006,))(input_tensor)

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

