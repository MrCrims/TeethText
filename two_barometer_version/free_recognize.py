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
                line = ser.readline().decode("utf-8")[:-1]
                baro_value = float(line)
                if baro_value > 150000 or baro_value < 50000:
                    continue
                with open(data_root_folder+'/'+"raw_baro.txt","a+") as f:
                    f.write(line+'\n')
            except:
                continue
            raw_baro.put(baro_value)
    except KeyboardInterrupt:
        pass

def process_2_monitor(class_num,sample_per_class,order_array,gesture_list,raw_baro,count_test,data_root_folder):
    random_time_array = np.array([1,2,3])
    np.random.shuffle(random_time_array)
    time_interval = random_time_array[0]
    all_baro = []
    gesture_unit = []

    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
        

    initial_baro_value = config_para['initial_baro_value']
    threshold_baro_energy = config_para['threshold_baro_energy']
    threshold_baro_length_min = config_para['threshold_baro_length_min']
    threshold_baro_length_max = config_para['threshold_baro_length_max']
    theshold_width = config_para['theshold_width']
    buffer_size = config_para['buffer_size']
    extend_data_length = config_para['extend_data_length']
    norm_type = config_para['norm_type']

    count_point = 0
    count_segment = 0
    count_recognize = 0
    flag_pos = 0
    current_gesture_id = int(order_array[int(count_test.value)])

    new_clf = joblib.load(config_para['model_path']) 

    print('Please start..................')
    t4 = time.time()
    while True:
        if raw_baro.empty() == False:
            count_point += 1
            baro_point = raw_baro.get()
            all_baro.append(baro_point)
            if abs(baro_point-initial_baro_value)>threshold_baro_energy:
                if flag_pos == 0:
                    t4 = time.time()
                    if len(all_baro) > buffer_size:
                        gesture_unit = all_baro[-buffer_size:]
                    else:
                        gesture_unit = all_baro
                    flag_pos = 1
                    start_pos = count_point
                else:
                    gesture_unit.append(baro_point)
                    flag_pos = count_point
            elif count_point - flag_pos < theshold_width and count_point - flag_pos > 0 and flag_pos > 0:
                gesture_unit.append(baro_point)
            elif count_point - flag_pos > 0 and flag_pos > 0:
                t0 = time.time()
                print('collection time: ', t0-t4)
                if len(gesture_unit) > threshold_baro_length_min and len(gesture_unit) < threshold_baro_length_max:
                    print('gesture length: ',len(gesture_unit))
                    t1 = time.time()
                    gesture_unit_data = np.array(gesture_unit)
                    np.savetxt(data_root_folder+'/segment_data/test/'+'%d_%d_%d.txt' % 
                        (count_segment, start_pos, current_gesture_id), gesture_unit_data, fmt='%.2f')
                    count_segment += 1
                    print('detected! Wait for next one............')
                    gesture_unit_data_extend = teethtext_module.data_extend(gesture_unit_data.reshape((len(gesture_unit_data), 1)),max_length=extend_data_length)
                    gesture_unit_data_extend = teethtext_module.norm_data(gesture_unit_data_extend,norm_type,initial_value=initial_baro_value)
                    gesture_unit_data_extend = gesture_unit_data_extend.reshape((1, len(gesture_unit_data_extend)))
                    predicted_gesture = new_clf.predict(gesture_unit_data_extend)[0]
                    print('Predict: ',gesture_list[int(predicted_gesture)])
                    t2 = time.time()
                    print('calculation time: ',t2-t1)
                    count_recognize += 1
                    print('recognition number: ',count_recognize)
                    print('-'*100)
                else:
                    print('invalid teeth gesture, the gesture length is: ', len(gesture_unit))
                gesture_unit = []
                flag_pos = 0
                
                

# next step
# 0. sometimes change very slow...时间主要卡在哪里了？
# 1. increase accuracy: a. change clearer teeth gesture, b. extract features from data and use SVM, c. standard template
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
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    gesture_list = config_para['gesture_list']
    class_num = len(gesture_list)
    sample_per_class = config_para['test_sample_num']
    order_array = teethtext_module.generate_id(class_num,sample_per_class)
    np.savetxt(data_root_folder+'/'+'order_array.txt',order_array.reshape(-1,1),fmt='%d')
    # print(order_array)

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

# 系统识别gesture的速度还可以接着优化