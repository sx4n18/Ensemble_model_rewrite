import numpy as np
import os

add_path_name = '/log_preprocess'
cur_dir = os.path.curdir

whole_path = cur_dir+add_path_name
if not(os.path.isdir(whole_path)):
    os.mkdir(whole_path)


for i in range(20):
    single_file_name = '/diagonal_'+str(i)+'.npy'
    new_single_file_name = '/log_data_preprocess_'+str(i)+'.npy'
    each_diagonal = np.load(cur_dir+single_file_name)
    new_log_diagonal = np.log(each_diagonal)
    processed_new_log_dia = np.nan_to_num(new_log_diagonal, neginf=0)
    np.save(whole_path+new_single_file_name,processed_new_log_dia)
