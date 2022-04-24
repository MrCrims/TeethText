 '''
Procedure:
Session 1: 5 samples for training
Session 2: 10 samples for testing
Session 3: 10 samples for cross validation

Notes:
1. Save all files and raw data
2. Whenever participants make a mistake, he/she should use finger gesture to tell the researcher
'''
import matplotlib.pyplot as plt
import teethtext_module
import numpy as np
from math import *
import serial, time, os, json
from multiprocessing import Process, Array, Value, Queue
import multiprocessing
import config_para

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
                baro_value = np.vstack((baro_data_side_1,baro_data_side_2)).T
                if baro_data_side_1 > 150000 or baro_data_side_1 < 50000 or baro_data_side_2 > 150000 or baro_data_side_2 < 50000:
                    continue
                with open(data_root_folder+'/'+"raw_baro.txt","a+") as f:
                    f.write(baro_data+'\n')
            except:
                continue
            raw_baro.put(baro_value)
    except KeyboardInterrupt:
        pass
def process_2_monitor(class_num,sample_per_class,order_array,gesture_list,raw_baro,count_train,data_root_folder,user_name):
    time_base = time.time()
    random_time_array = np.array([1,2,3])
    np.random.shuffle(random_time_array)
    time_interval = random_time_array[0]
    all_baro = np.empty([0,2])
    gesture_unit = np.empty([0,2])
    
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
        

    initial_baro_value = np.array(config_para['initial_baro_value'])
    threshold_baro_energy = np.array(config_para['threshold_baro_energy'])
    threshold_baro_length_min = config_para['threshold_baro_length_min']
    threshold_baro_length_max = config_para['threshold_baro_length_max']
    theshold_width = config_para['theshold_width']
    buffer_size = config_para['buffer_size']
    extend_data_length = config_para['extend_data_length']
    norm_type = config_para['norm_type']

    count_point = 0
    count_segment = 0
    flag_pos = 0
    flag_finish = 1
    current_gesture_id = int(order_array[int(count_train.value)])
    
    while count_train.value < class_num * sample_per_class or flag_finish == 0:
        if flag_pos == 0 and time.time() - time_base > time_interval and flag_finish == 1:
            current_gesture_id = int(order_array[int(count_train.value)])

            print('NO.%d: ' % int(count_train.value), gesture_list[current_gesture_id])
            count_train.value += 1
            flag_finish = 0
            random_time_array = np.array([1,2,3])
            np.random.shuffle(random_time_array)
            time_interval = random_time_array[0]
        if raw_baro.empty() == False:
            count_point += 1
            # print(raw_baro.qsize())
            baro_point = raw_baro.get()
            all_baro = np.append(all_baro,baro_point,axis=0)
            judge = abs(baro_point-initial_baro_value)>threshold_baro_energy
            if judge.any():
               # print("delta:",abs(baro_point-initial_baro_value))
               # print("flag_pos1:",flag_pos)
               # print('count_point:', count_point)
                if flag_pos == 0:
                    if len(all_baro) > buffer_size:
                        gesture_unit = all_baro[-buffer_size:]
                    else:
                        gesture_unit = all_baro
                    flag_pos = count_point
                    start_pos = count_point
                else:
                    #gesture_unit.append(baro_point)
                    gesture_unit = np.append(gesture_unit,baro_point,axis=0)
                    flag_pos = count_point
            elif count_point - flag_pos < theshold_width and count_point - flag_pos > 0 and flag_pos > 0:
                #gesture_unit.append(baro_point)
                gesture_unit = np.append(gesture_unit, baro_point, axis=0)
                #print("flag_pos2:", flag_pos)
            elif count_point - flag_pos > 0 and flag_pos > 0:
                if len(gesture_unit) > threshold_baro_length_min and len(gesture_unit) < threshold_baro_length_max:
                    t1 = time.time()
                    np.savetxt(data_root_folder+'/segment_data/train/'+'%d_%d_%d.txt' % 
                        (count_segment, start_pos, current_gesture_id), np.array(gesture_unit), fmt='%.2f')
                    count_segment += 1
                    print('detected! Wait for next one............')
                    print('-'*80)
                   # print("flag_pos1:", flag_pos)
                   # print('count_point:', count_point)
                    # print(raw_baro.qsize())
                    t2 = time.time()
                    # print('processing time: ',t2-t1)
                    flag_finish = 1
                    time_base = time.time()
                else:
                    print('invalid teeth gesture')
                gesture_unit = np.empty([0,2])
                #print('000')
                flag_pos = 0
                
    teethtext_module.process_file_train(data_root_folder+'/segment_data/train',extend_data_length,norm_type=norm_type,initial_value=initial_baro_value)
    teethtext_module.train_model('model/train_data.npy','model/train_label.npy',user_name)
                

# next step
# 0. sometimes change very slow...
# 1. increase accuracy: a. change clearer teeth gesture, b. extract features from data and use SVM, c. standard template
# 2. increase the speed and efficiency of segmentation part
# 3. distinguish noise or teeth gesture
# 4. typing implementation

if __name__ == '__main__':
    # main code
    # baro sample rate: 115-125 Hz
    # 用户需要保持动作的一致性，即嘴一直闭着，不要张开，不能有时张开，有时闭着
    user_name = input("please input your name: ")
    config_para.dict_to_json_write_file(user_name)
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    data_root_folder = 'user_data/'+user_name+'_'+date
    os.mkdir(data_root_folder)
    os.mkdir(data_root_folder+'/segment_data')
    os.mkdir(data_root_folder+'/segment_data/train')
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    gesture_list = config_para['gesture_list']
    class_num = len(gesture_list)
    sample_per_class = config_para['train_sample_num']
    order_array = teethtext_module.generate_id(class_num,sample_per_class)
    np.savetxt(data_root_folder+'/'+'order_array.txt',order_array.reshape(-1,1),fmt='%d')
    print(order_array)

    raw_baro = Queue()
    count_train = Value('d',0)

    p1 = Process(target=process_1_collect_baro,args=(class_num,sample_per_class,raw_baro,count_train,
        data_root_folder,))
    p2 = Process(target=process_2_monitor,args=(class_num,sample_per_class,order_array,gesture_list,
        raw_baro,count_train,data_root_folder,user_name,))
    # start process
    p1.start()
    p2.start()
    # finish process
    p2.join()
    p1.terminate()
