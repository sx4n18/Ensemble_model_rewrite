import numpy as np
import os

add_path_name = '/log_preprocess'
cur_dir = os.path.curdir

whole_path = cur_dir+add_path_name
if not(os.path.isdir(whole_path)):
    os.mkdir(whole_path)

sample_num = 9000
channel_num = 1024
ensemble_num = 20

file_name = '/log_plus_one_raw_data.npy'
extra_path = '/log_plus_1_subtraction_prepro'
big_logged_dat = np.load(whole_path+file_name)
big_processed_data = []

def create_ratioed_matrix(single_sample):
    new_matrix = np.zeros((channel_num, channel_num))
    for row in range(channel_num):
            new_matrix[row] = single_sample-single_sample[row]

    return new_matrix

def get_the_diagonal(input_matrix,line):
    new_diagonal_elements = np.zeros((channel_num-line))
    for index in range(channel_num-line):
        new_diagonal_elements[index] = input_matrix[index,index+line]

    return new_diagonal_elements

## create a big list of empty arrays for each ensembled single ANN
for ensemble_index in range(ensemble_num):
    big_processed_data.append(np.zeros((sample_num, channel_num-ensemble_index-1)))

for sample_index in range(sample_num):
    single_raw_sample = big_logged_dat[sample_index]
    one_sample_ratio = create_ratioed_matrix(single_raw_sample)
    for ensemble_index in range(ensemble_num):
        diagonal_for_this_ensemble = get_the_diagonal(one_sample_ratio,ensemble_index+1)
        big_processed_data[ensemble_index][sample_index] = diagonal_for_this_ensemble

for index in range(ensemble_num):
    try:
        np.save('./log_preprocess/log_plus_1_subtraction_prepro/diagonal_preprocess_log_'+str(index)+'.npy', big_processed_data[index])
    except:
        os.mkdir(whole_path+extra_path)
        np.save('./log_preprocess/log_plus_1_subtraction_prepro/diagonal_preprocess_log_' + str(index) + '.npy',
                big_processed_data[index])





