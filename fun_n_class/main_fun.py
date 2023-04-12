import numpy as np
from tensorflow import keras
from keras.losses import categorical_crossentropy


##
# Funciton definition of this rewrite project
##

total_sample_number = 9000
total_testing_number = int(0.15*total_sample_number)




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
    final_prediction_after_voting = np.argmax(overall_odds_from_each_ANN,axis=1)

    return final_prediction_after_voting


def sub_cls_to_sequential(old_sub_cls_model, current_index):
    variables = old_sub_cls_model.variables
    new_sequential = keras.Sequential([
        keras.layers.Dense(40, activation='sigmoid'),
        keras.layers.Dense(18, activation='softmax')
    ])
    new_sequential.build(input_shape=(None, 1023-current_index))
    new_sequential.set_weights(variables)
    new_sequential.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


    return new_sequential