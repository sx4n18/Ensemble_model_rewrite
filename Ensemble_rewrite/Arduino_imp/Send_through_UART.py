import serial
import numpy as np
import time
from sklearn.model_selection import train_test_split


#################################################################################
# This script aims to automate the testing to send through data from UART.
# And receive the inferencing result, speedy test version
#################################################################################

def begin_sending():
    port.write(str.encode(Starter_str))
    time.sleep(0.0002)


def for_loop_send_int(flat_sample):
    for number in flat_sample:
        port.write(str.encode(str(number) + ' '))
        time.sleep(0.0002)


raw_data_9000_by_1024 = np.load('../data_set/original_histogram.npy')
label = np.load('../data_set/label.npy') - 1
Starter_str = "C"
total_sample = 1350
test_number = 0
correct_num = 0
chosen_combo_5 = np.array([5, 8, 11, 17, 19])
x_test_lst = []
for item in chosen_combo_5:
    raw_data_this_dia = np.load(
        '../data_set/log_preprocess/log_plus_1_subtraction_prepro/diagonal_preprocess_log_' + str(item) + '.npy')
    X_train, X_test, y_train, y_test = train_test_split(raw_data_this_dia, label, test_size=0.15,
                                                        random_state=42, shuffle=True)
    x_test_lst.append(X_test)

X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(raw_data_9000_by_1024, label, test_size=0.15,
                                                                    random_state=42, shuffle=True)

merged_x_test = np.concatenate(x_test_lst, axis=1)

if merged_x_test.shape[1] != 5055:
    print("data shape wrong!")

try:
    port = serial.Serial('/dev/cu.usbmodem11401', 9600, timeout=2)
except serial.SerialException:
    try:
        port = serial.Serial('/dev/cu.usbmodem1301', 19200, timeout=2)
    except serial.SerialException:
        port = serial.Serial('/dev/cu.usbmodem1401', 9600, timeout=2)

# Since there are 4 lines printed out through UART,
# the welcoming line will be collected and printed 4 times
for i in range(4):
    welcoming_line = port.readline()
    print(welcoming_line.decode())

process_beginning_time = time.time()

while test_number < total_sample:
    sending_indicator = port.readline()
    start_time = time.time()
    while str(sending_indicator.decode()) != "Waiting 4 'C'\r\n":
        sending_indicator = port.readline()
        end_time = time.time()
        if end_time - start_time > 3:
            raise TimeoutError
    # send the big C to kickstart data transfer
    begin_sending()
    # send the sample each by each
    sample = X_raw_test[test_number].flatten()
    for_loop_send_int(sample)
    # printing the result
    print("Actual label:", y_raw_test[test_number])
    predicted_lb = port.readline()
    print("Predicted:", predicted_lb.decode())
    # accuracy checking
    if y_raw_test[test_number] == int(predicted_lb.decode()):
        correct_num += 1
    test_number += 1
process_ending_time = time.time()

print("Overall accuracy:", correct_num * 100 / total_sample, "%")
time_lapse = process_ending_time - process_beginning_time

print("Overall time lapse: {time_lapse:.2} seconds".format(time_lapse=time_lapse))
