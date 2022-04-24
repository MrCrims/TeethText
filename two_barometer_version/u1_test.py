import matplotlib.pyplot as plt
import teethtext_module
import numpy as np
from math import *
import serial, time, os, json
from multiprocessing import Process, Array, Value, Queue
import multiprocessing
import joblib


def process_1_collect_baro(class_num,sample_per_class,raw_baro,count_train,data_root_folder):
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    port_position = config_para['serial_port_pos']
    ser = serial.Serial(port_position, 9600)

    try:
        while True:
            try:
                baro_data = ser.readline().decode("utf-8")
                baro_data_side_1 = float(baro_data.split(',')[0])
                baro_data_side_2 = float(baro_data.split(',')[1].split('\r')[0])
                baro_value = np.vstack((baro_data_side_1, baro_data_side_2)).T
                if baro_data_side_1 > 150000 or baro_data_side_1 < 50000 or baro_data_side_2 > 150000 or baro_data_side_2 < 50000:
                    continue
                with open(data_root_folder + '/' + "raw_baro.txt", "a+") as f:
                    f.write(baro_data + '\n')
            except:
                continue
            raw_baro.put(baro_value)
    except KeyboardInterrupt:
        pass

def process_2_monitor(class_num,sample_per_class,order_array,gesture_list,raw_baro,count_test,data_root_folder):
    '''
    def gesture_filter(gesture_id, gesture_data, gesture_max_min_interval_threshold = 20):
        # Single-Right-Click & Single-Down-Click
        if gesture_id == 1 or gesture_id == 4:
            if abs(np.argmax(gesture_data) - np.argmin(gesture_data)) < gesture_max_min_interval_threshold:
                print('Short Interval...')
                return 4 # Single-Down-Click
            else:
                print('Long Interval...')
                return 1 # Single-Right-Click
        else:
            return gesture_id
    '''
    time_base = time.time()
    random_time_array = np.array([1,2,3])
    np.random.shuffle(random_time_array)
    time_interval = random_time_array[0]
    all_baro = np.empty([0, 2])
    gesture_unit = np.empty([0, 2])
    
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)

    count = 0
    time1 = time.time()
    init_baro_value_test = 0
    while time.time() - time1 < 4:
        if 2 < time.time() - time1 < 3:
            try:
                baro_data = raw_baro.get()
                # print(baro_data)
                init_baro_value_test += baro_data
                count += 1
            except:
                continue

    init_baro_value_test = init_baro_value_test/count
    initial_baro_value_train = config_para['initial_baro_value']
    init_delta = init_baro_value_test-initial_baro_value_train
    #threshold_baro_energy = config_para['threshold_baro_energy']
    baro_coefficient = config_para['baro_coefficient']
    threshold_baro_energy = init_baro_value_test*baro_coefficient
    threshold_baro_length_min = config_para['threshold_baro_length_min']
    threshold_baro_length_max = config_para['threshold_baro_length_max']
    theshold_width = config_para['theshold_width']
    buffer_size = config_para['buffer_size']
    extend_data_length = config_para['extend_data_length']
    norm_type = config_para['norm_type']
    gesture_max_min_interval_threshold = config_para['gesture_max_min_interval_threshold']

    count_point = 0
    count_segment = 0
    count_recognize = 0
    flag_pos = 0
    flag_finish = 1
    current_gesture_id = int(order_array[int(count_test.value)])

    new_clf = joblib.load(config_para['model_path']) 

    tn = time.time()
    while count_test.value < class_num * sample_per_class or flag_finish == 0:
        if flag_pos == 0 and time.time() - time_base > time_interval and flag_finish == 1:
            current_gesture_id = int(order_array[int(count_test.value)])
            print('NO.%d: ' % int(count_test.value), gesture_list[current_gesture_id])
            count_test.value += 1
            flag_finish = 0
            random_time_array = np.array([1,2,3])
            np.random.shuffle(random_time_array)
            time_interval = random_time_array[0]
        if raw_baro.empty() == False:
            count_point += 1
            baro_point = raw_baro.get()
            #all_baro.append(baro_point-init_delta)
            all_baro = np.append(all_baro, baro_point-init_delta, axis=0)
            judge = abs(baro_point-init_baro_value_test)>threshold_baro_energy
            if judge.any():
                if flag_pos == 0:
                    tn = time.time()
                    if len(all_baro) > buffer_size:
                        gesture_unit = all_baro[-buffer_size:]
                    else:
                        gesture_unit = all_baro
                    flag_pos = count_point
                    start_pos = count_point
                else:
                    #gesture_unit.append(baro_point-init_delta)
                    gesture_unit = np.append(gesture_unit,baro_point-init_delta,axis=0)
                    flag_pos = count_point
            elif theshold_width > count_point - flag_pos > 0 and flag_pos > 0:
                #gesture_unit.append(baro_point-init_delta)
                gesture_unit = np.append(gesture_unit, baro_point - init_delta, axis=0)
            elif count_point - flag_pos > 0 and flag_pos > 0:
                if threshold_baro_length_min < len(gesture_unit) < threshold_baro_length_max:
                    t1 = time.time()
                    gesture_unit_data = gesture_unit
                    np.savetxt(data_root_folder+'/segment_data/test/'+'%d_%d_%d.txt' % 
                        (count_segment, start_pos, current_gesture_id), gesture_unit_data, fmt='%.2f')
                    count_segment += 1
                    print('detected! Wait for next one............')

                    # print(raw_baro.qsize())
                    # print('-'*80)
                    gesture_unit_data_extend = teethtext_module.data_extend(gesture_unit_data,max_length=extend_data_length)
                    gesture_unit_data_extend = teethtext_module.norm_data(gesture_unit_data_extend,norm_type,initial_value=init_baro_value_test)
                    #gesture_unit_data_extend = gesture_unit_data_extend.reshape((1, len(gesture_unit_data_extend)))
                    gesture_unit_data_extend = gesture_unit_data_extend.reshape((1,extend_data_length,2))
                    predicted_gesture = new_clf.predict(gesture_unit_data_extend)[0]
                    # predicted_gesture = gesture_filter(predicted_gesture, gesture_unit_data_extend, gesture_max_min_interval_threshold)

                    with open(data_root_folder+'/'+"ground_truth.txt","a+") as f3:
                        f3.write(str(int(current_gesture_id))+'\n')
                    with open(data_root_folder+'/'+"predict_result.txt","a+") as f4:
                        f4.write(str(int(predicted_gesture))+'\n')
                    with open(data_root_folder+'/'+"gesture_time.txt","a+") as f5:
                        f5.write(str(t1 - tn)+'\n')
                    if predicted_gesture == current_gesture_id:
                        print('Right Gesture! ',gesture_list[int(predicted_gesture)])
                        count_recognize += 1
                    else:
                        print('Wrong! Predict: ',gesture_list[int(predicted_gesture)],'Truth: ',gesture_list[current_gesture_id])
                        ttt = time.time()
                    print('\n')
                    flag_finish = 1
                    t2 = time.time()
                    # print('calculation time: ', t2-t1)
                    time_base = time.time()

                else:
                    print('invalid teeth gesture')
                gesture_unit = []
                flag_pos = 0
                
    print('Final Accuracy: ',count_recognize/len(order_array))
                

# next step
# 1. see the accuracy of dtw + knn
# 2. increase the speed and efficiency of segmentation part
# 3. distinguish noise or teeth gesture
# 4. typing implementation

if __name__ == '__main__':
    # main code
    # baro sample rate: 115-125 Hz
    # 用户需要保持动作的一致性，即嘴一直闭着，不要张开，不能有时张开，有时闭着
    user_name = input("please input your name: ")
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    data_root_folder = 'user_data/'+user_name+'_'+date
    os.mkdir(data_root_folder)
    os.mkdir(data_root_folder+'/segment_data')
    os.mkdir(data_root_folder+'/segment_data/test')
    # gesture_file = open('gesture_list.txt', 'r')
    # gesture_list = gesture_file.read().split(',')
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    gesture_list = config_para['gesture_list']
    class_num = len(gesture_list)
    sample_per_class = config_para['test_sample_num']
    order_array = teethtext_module.generate_id(class_num,sample_per_class)
    np.savetxt(data_root_folder+'/'+'order_array.txt',order_array.reshape(-1,1),fmt='%d')
    print(order_array)

    raw_baro = Queue()
    count_test = Value('d',0)

    p1 = Process(target=process_1_collect_baro,args=(class_num,sample_per_class,raw_baro,count_test,
        data_root_folder,))
    p2 = Process(target=process_2_monitor,args=(class_num,sample_per_class,order_array,gesture_list,
        raw_baro,count_test,data_root_folder,))
    # start process
    p1.start()    
    p2.start()
    # finish process
    p2.join()
    p1.terminate()
