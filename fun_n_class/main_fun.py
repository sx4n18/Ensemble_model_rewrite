import numpy as np
from tensorflow import keras
from keras.losses import categorical_crossentropy
import tensorflow as tf

##
# Funciton definition of this rewrite project
##

total_sample_number = 9000
total_testing_number = int(0.15 * total_sample_number)


def major_hardvote(all_diagonal_prediction):
    '''
    :param all_diagonal_prediction: a 3 dimensional array that contains the final results from the input
    :return: the voting results
    '''
    final_class_of_each_ANN = np.argmax(all_diagonal_prediction, axis=2)
    final_prediction_after_voting = np.zeros(total_testing_number, 'int')
    for sample_index in range(total_testing_number):
        bin_count = np.bincount(final_class_of_each_ANN[:, sample_index])
        final_prediction_after_voting[sample_index] = np.argmax(bin_count)

    return final_prediction_after_voting


def major_softvote(all_diagona_prediction):
    overall_odds_from_each_ANN = np.sum(all_diagona_prediction, axis=0)
    final_prediction_after_voting = np.argmax(overall_odds_from_each_ANN, axis=1)

    return final_prediction_after_voting


def sub_cls_to_sequential(old_sub_cls_model, current_index, original=True):
    variables = old_sub_cls_model.variables
    if original:
        act_option = 'sigmoid'
    else:
        act_option = 'relu'
    new_sequential = keras.Sequential([
        keras.layers.Input(shape=(1023 - current_index,)),
        keras.layers.Dense(40, activation=act_option),
        keras.layers.Dense(18, activation='softmax')
    ])
    # new_sequential.build(input_shape=(None, 1023-current_index))
    new_sequential.set_weights(variables)
    new_sequential.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return new_sequential


def tflite_prediction_with_int8_input(tf_lite_model, total_testing_number, input_num_this_dia, total_cls_num, Test_X):
    interpreter = tf_lite_model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('Input shape:', input_details[0]['shape'])
    print('Output shape:', output_details[0]['shape'])

    interpreter.resize_tensor_input(input_details[0]['index'], (total_testing_number, input_num_this_dia))
    interpreter.resize_tensor_input(output_details[0]['index'], (total_testing_number, total_cls_num))
    interpreter.allocate_tensors()

    print('After the resize...')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('Input shape:', input_details[0]['shape'])
    print('Output shape:', output_details[0]['shape'])

    # quantise the input test images

    test_img = Test_X / input_details[0]['quantization_parameters']['scales'] + \
               input_details[0]['quantization_parameters']['zero_points']
    test_img = np.array(test_img, dtype=np.int8)

    interpreter.set_tensor(input_details[0]['index'], test_img)
    interpreter.invoke()
    model_prediction = interpreter.get_tensor(output_details[0]['index'])

    return model_prediction


def give_final_result_from_ensemble(concated_result, total_cls_num):
    num_of_nets = int(concated_result.shape[1]/total_cls_num)
    each_prediction = np.zeros((num_of_nets, concated_result.shape[0], total_cls_num))
    for each_net in range(num_of_nets):
        each_prediction[each_net] = concated_result[:, 18*each_net:18*(each_net+1)]

    final_result = major_hardvote(each_prediction)

    return final_result


def tflite_prediction_with_normal_input(tf_lite_model, total_testing_number, input_num_this_dia, total_cls_num, Test_X):
    interpreter = tf_lite_model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('Input shape:', input_details[0]['shape'])
    print('Output shape:', output_details[0]['shape'])

    interpreter.resize_tensor_input(input_details[0]['index'], (total_testing_number, input_num_this_dia))
    interpreter.resize_tensor_input(output_details[0]['index'], (total_testing_number, total_cls_num))
    interpreter.allocate_tensors()

    print('After the resize...')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('Input shape:', input_details[0]['shape'])
    print('Output shape:', output_details[0]['shape'])

    # quantise the input test images

    test_img = np.array(Test_X, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_img)
    interpreter.invoke()
    model_prediction = interpreter.get_tensor(output_details[0]['index'])

    return model_prediction