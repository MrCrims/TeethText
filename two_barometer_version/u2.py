import serial, os, time, random, ctypes, sys, json
import numpy as nps
from multiprocessing import Process, Array, Value, Queue
import multiprocessing
import pandas as pd
import matplotlib as mp 
mp.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import teethtext_module
from math import *
import joblib
import numpy as np
from random import randint, choice

def process_0_collect_baro(raw_baro,data_root_folder):
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

def process_1_gesture_detection(raw_baro,data_root_folder,user_text_queue,refresh_flag,next_phrase_flag,auto_text_queue,target_text_queue,input_number_flag,input_number_num,input_gesture_id):
    def get_phrase(dir):
        phrase_set = pd.read_csv(dir)
        phrase_set = np.array(phrase_set)
        return phrase_set

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

    count_point = 0
    count_segment = 0
    flag_pos = 0

    new_clf = joblib.load(config_para['model_path']) 

    gesture_list = config_para['gesture_list']

    Teeth_Gesture_Select_Set = gesture_list[0:4]
    Teeth_Gesture_Delete_Action = gesture_list[5]
    Teeth_Gesture_Jump_Action = gesture_list[4]
    # Teeth_Gesture_Jump_Action_Set = ['Single_Back_Click','Double_Back_Click','Triple_Back_Click']
    Teeth_Gesture_Next_Action = gesture_list[6]

    ### load word prob dictionary
    with open('text_entry_utils/word_prob_15000.json', 'r') as f1:
        word_prob_dict = json.load(fp=f1)
    with open('text_entry_utils/spatial_model_4_group_dict.json', 'r') as f2:
        emission_matrix_dict = json.load(fp=f2)

    jump_flag = 0
    phrase_lock = 0
    delete_lock = 0
    gesture_num = 0
    word_num_per_phrase = 0
    user_self_input_num_per_word = 0
    count_phrase = 0
    phrase_set = get_phrase(config_para['phrase_path'])
    norm_type = config_para['norm_type']

    auto_word_set = []
    user_word_set = []
    input_number_set = []
    print('initializing ...')
    # time.sleep(3)
    temp = target_text_queue.get()
    target_text_queue.put(phrase_set[0][0])
    target_word_set = target_text_queue.get()
    target_text_queue.put(target_word_set)

    timer_1 = time.time()
    timer_2 = time.time()

    word_time_1 = time.time()
    word_time_2 = time.time()

    t1 = time.time()
    t2 = time.time()
    t3 = time.time()

    auto_predict_time_1 = time.time()
    auto_predict_time_2 = time.time()

    def gesture_to_number(teeth_gesture):
        if teeth_gesture in Teeth_Gesture_Select_Set:
            return str(Teeth_Gesture_Select_Set.index(teeth_gesture) + 1)
        else:
            print('invalid teeth gesture...')
            return '0'

    while True:
        if raw_baro.empty() == False:
            count_point += 1
            baro_point = raw_baro.get()
            all_baro.append(baro_point)
            if abs(baro_point-initial_baro_value)>threshold_baro_energy:
                if flag_pos == 0:
                    t3 = time.time()
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
                if len(gesture_unit) > threshold_baro_length_min and len(gesture_unit) < threshold_baro_length_max:
                    t1 = time.time()
                    print('*'*100)
                    print('gesture collection time: ', t1-t3)
                    gesture_unit_data = np.array(gesture_unit)
                    np.savetxt(data_root_folder+'/segment_data/test/'+'%d_%d.txt' % (count_segment, start_pos), gesture_unit_data, fmt='%.2f')
                    count_segment += 1
                    print('detected! Wait for next one............')
                    gesture_unit_data_extend = teethtext_module.data_extend(gesture_unit_data.reshape((len(gesture_unit_data), 1)),max_length=extend_data_length)
                    gesture_unit_data_extend = teethtext_module.norm_data(gesture_unit_data_extend,norm_type,initial_value=initial_baro_value)
                    gesture_unit_data_extend = gesture_unit_data_extend.reshape((1, len(gesture_unit_data_extend)))
                    predicted_gesture_index = new_clf.predict(gesture_unit_data_extend)[0]
                    predicted_gesture = gesture_list[int(predicted_gesture_index)]
                    print('Predict: ', predicted_gesture)
                    t2 = time.time()
                    print('gesture calculation time: ',t2-t1)
                    gesture_num += 1
                    input_gesture_id.value = int(predicted_gesture_index)
                    if gesture_num == 1:
                        word_time_1 = time.time()
                        timer_1 = time.time()
                        print('start now !!!!!','.'*80)
                    if predicted_gesture in Teeth_Gesture_Select_Set and gesture_num > 1 and phrase_lock == 0:
                        if jump_flag != 0 and len(user_word_set) != 0:
                            target_word_set = target_text_queue.get()
                            target_text_queue.put(target_word_set)
                            if delete_lock == 0:
                                with open(data_root_folder+'/'+"user_entry_word.txt","a+") as f3:
                                    f3.write(str(user_word_set[-2])+'\n')
                                with open(data_root_folder+'/'+"user_entry_word_time.txt","a+") as f4:
                                    f4.write(str(word_time_2-word_time_1)+'\n')
                                with open(data_root_folder+'/'+"truth_text.txt","a+") as f5:
                                    f5.write(str(target_word_set.split()[word_num_per_phrase])+'\n')
                                with open(data_root_folder+'/'+"user_self_input_word_length.txt","a+") as f6:
                                    f6.write(str(user_self_input_num_per_word)+'\n')
                                with open(data_root_folder+'/'+"user_input_gesture_set.txt","a+") as f7:
                                    f7.write(''.join(input_number_set_temp)+'\n')
                                print('User Input Word: ',user_word_set[-2])
                                print('Truth Word: ',target_word_set.split()[word_num_per_phrase])
                                print('Single Word Input Time: ',word_time_2-word_time_1)
                                word_time_1 = word_time_2
                            word_num_per_phrase += 1
                            print('word_num_per_phrase: ', word_num_per_phrase)
                            jump_flag = 0
                            user_self_input_num_per_word = 0
                        elif jump_flag != 0:
                            word_num_per_phrase += 1
                            jump_flag = 0
                            user_self_input_num_per_word = 0
                        input_number = gesture_to_number(predicted_gesture)
                        input_number_set.append(input_number)
                        input_number_set_temp = input_number_set
                        input_number_num.value += 1
                        user_self_input_num_per_word += 1
                        print('input_number_set: ',input_number_set)
                        auto_predict_time_1 = time.time()
                        auto_word_set = teethtext_module.calculate_prob(input_number_set, word_prob_dict, emission_matrix_dict)
                        auto_predict_time_2 = time.time()
                        print('auto predict word time: ', auto_predict_time_2 - auto_predict_time_1)
                        candidate_top_three = ' '.join(auto_word_set)
                        temp = auto_text_queue.get()
                        auto_text_queue.put(candidate_top_three)
                        input_number_flag.value = int(input_number)
                        refresh_flag.value = 1
                        delete_lock = 0
                        print('target_word_set: ', target_word_set)
                        print('auto_word_set: ', auto_word_set)
                        print('user_word_set: ', user_word_set)

                    elif predicted_gesture == Teeth_Gesture_Delete_Action and gesture_num > 1:
                        if jump_flag == 0:
                            input_number_set = input_number_set[0:-1]
                            input_number_set_temp = input_number_set
                            input_number_num.value -= 1
                            if len(input_number_set) == 0:
                                auto_word_set = []
                            else:
                                auto_predict_time_1 = time.time()
                                auto_word_set = teethtext_module.calculate_prob(input_number_set, word_prob_dict, emission_matrix_dict)
                                auto_predict_time_2 = time.time()
                                ('auto predict word time: ', auto_predict_time_2 - auto_predict_time_1)
                            candidate_top_three = ' '.join(auto_word_set)
                            temp = auto_text_queue.get()
                            auto_text_queue.put(candidate_top_three)
                            refresh_flag.value = 1
                            print('target_word_set: ', target_word_set)
                            print('auto_word_set: ', auto_word_set)
                            print('user_word_set: ', user_word_set)
                            print('input_number_set: ',input_number_set)
                            print('cancel the last gesture')
                        elif len(user_word_set) > 0:
                            del user_word_set[-1]
                            del user_word_set[-1]
                            phrase_lock = 0
                            word_num_per_phrase -= 1
                            input_number_set = []
                            input_number_num.value = 0
                            auto_word_set = []
                            candidate_top_three = ' '
                            temp = auto_text_queue.get()
                            auto_text_queue.put(candidate_top_three)
                            temp = user_text_queue.get()
                            user_text_queue.put(''.join(user_word_set))
                            refresh_flag.value = 1
                            print('target_word_set: ', target_word_set)
                            print('auto_word_set: ', auto_word_set)
                            print('user_word_set: ', user_word_set)
                            print('input_number_set: ',input_number_set)
                            print('cancel the last word')
                            jump_flag = 1
                            delete_lock = 1

                    elif predicted_gesture == Teeth_Gesture_Jump_Action and gesture_num > 1 and len(auto_word_set) != 0:
                        
                        timer_2 = time.time()
                        word_time_2 = time.time()
                        jump_flag += 1
                        if jump_flag == 1:
                            user_word_set.append(auto_word_set[jump_flag-1])
                            user_word_set.append(' ')
                            input_number_set = []
                            input_number_num.value = 0
                            
                        elif jump_flag > 1:
                            user_word_set[-2] = auto_word_set[int((jump_flag-1)%3)]
                            input_number_set = []
                            input_number_num.value = 0
                        
                        if len(user_word_set) == 2 * len(target_word_set.split()):
                            phrase_lock = 1

                        temp = user_text_queue.get()
                        user_text_queue.put(''.join(user_word_set))
                        refresh_flag.value = 1
                        print('target_word_set: ', target_word_set)
                        print('auto_word_set: ', auto_word_set)
                        print('user_word_set: ', user_word_set)
                        print('input_number_set: ', input_number_set)

                    elif predicted_gesture == Teeth_Gesture_Next_Action and gesture_num > 1:
                        phrase_lock = 0
                        print('word_num_per_phrase: ',word_num_per_phrase)
                        if word_num_per_phrase + 1 == len(target_word_set.split()):
                            with open(data_root_folder+'/'+"user_entry_word.txt","a+") as f3:
                                f3.write(str(user_word_set[-2])+'\n')
                            with open(data_root_folder+'/'+"user_entry_word_time.txt","a+") as f4:
                                f4.write(str(word_time_2-word_time_1)+'\n')
                            with open(data_root_folder+'/'+"truth_text.txt","a+") as f5:
                                f5.write(str(target_word_set.split()[word_num_per_phrase])+'\n')
                            with open(data_root_folder+'/'+"user_self_input_word_length.txt","a+") as f6:
                                f6.write(str(user_self_input_num_per_word)+'\n')
                            with open(data_root_folder+'/'+"user_input_gesture_set.txt","a+") as f7:
                                f7.write(''.join(input_number_set_temp)+'\n')
                            with open(data_root_folder+'/'+"user_entry_phrase_time.txt","a+") as f8:
                                f8.write(str(timer_2-timer_1)+'\n')
                            print('User Input Word: ',user_word_set[-2])
                        
                            print('Truth Word: ',target_word_set.split()[word_num_per_phrase])
                            print('Single Word Input Time: ',word_time_2-word_time_1)
                            print('phrase input time: ',timer_2-timer_1)
                            user_self_input_num_per_word = 0
                            next_phrase_flag.value = 1
                            count_phrase = count_phrase + 1
                            temp = target_text_queue.get()
                            target_text_queue.put(phrase_set[count_phrase][0])
                            word_num_per_phrase = 0
                            user_word_set = []
                            auto_word_set = []
                            input_number_set = []
                            input_number_num.value = 0
                            target_word_set = target_text_queue.get()
                            target_text_queue.put(target_word_set)
                            word_time_1 = time.time()
                            print('start counting time...','.'*80)
                            print('word_num_per_phrase: ', word_num_per_phrase)
                            jump_flag = 0
                            timer_1 = time.time()
                            refresh_flag.value = 1
                            gesture_unit = []
                            flag_pos = 0
                            print('start counting time...','.'*80)
                        else:
                            gesture_unit = []
                            flag_pos = 0
                else:
                    print('invalid teeth gesture')
                gesture_unit = []
                flag_pos = 0

def process_2_GUI(user_text_queue,refresh_flag,next_phrase_flag,auto_text_queue,target_text_queue,input_number_flag,input_number_num,input_gesture_id):
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    
    gesture_list = config_para['gesture_list']

    fig, ax = plt.subplots(figsize=(6,4))
    width, height = 16, 16
    ax.set(xlim=[0, width], ylim=[0, height]) # Or use "ax.axis([x0,x1,y0,y1])"
    # before use background = fig.canvas.copy_from_bbox(ax.bbox), you must run: fig.canvas.draw()
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    x0 = np.linspace(2,8,10)
    y0 = np.ones(10) * 2

    # count_phrase = 0
    next_phrase_flag.value = 0
    # phrase_set = get_phrase('phrases_set_session_2.txt')
    # np.random.seed(5)
    # np.random.shuffle(phrase_set)

    def update():
        if refresh_flag.value == 1:
            if next_phrase_flag.value != 1:
                user_text = user_text_queue.get()
                user_text_queue.put(user_text)
                auto_text = auto_text_queue.get()
                auto_text_queue.put(auto_text)
                target_text = target_text_queue.get()
                target_text_queue.put(target_text)
                print('user_text: ', user_text, 'auto_text: ', auto_text)
                refresh_flag.value = 0
            else:
                target_text = target_text_queue.get()
                target_text_queue.put(target_text)
                auto_text = ''
                user_text = ''
                temp = user_text_queue.get()
                user_text_queue.put(user_text)
                temp = auto_text_queue.get()
                auto_text_queue.put(auto_text)
                refresh_flag.value = 0
                next_phrase_flag.value = 0
                print('next phrase ---------------------------------------- next one : ',target_text)
        else:
            user_text = user_text_queue.get()
            user_text_queue.put(user_text)
            auto_text = auto_text_queue.get()
            auto_text_queue.put(auto_text)
            target_text = target_text_queue.get()
            target_text_queue.put(target_text)

        fig.canvas.restore_region(background)
        start = time.time()

        ax.draw_artist(ax.plot(x0+7*((input_number_flag.value + 1) % 2),y0-1.5*(int((input_number_flag.value + 1) / 2) - 1))[0])
        ax.draw_artist(ax.text(2,6,'Target: '+'\n\n\n\n'+'Auto-Complete: '+
                                                            '\n\n'+'Input Number: '+'\n\n'+'Writing: '))
        ax.draw_artist(ax.text(7,4.5,gesture_list[int(input_gesture_id.value)],fontweight='light'))
        ax.draw_artist(ax.text(8,7.5,str(int(input_number_num.value)),fontweight='light'))
        ax.draw_artist(ax.text(2,1,'A    B    C    D    E    F    G   ||   H    I      J     K    L    M    N' +
                                '\n\n' +  '      O    P    Q    R    S    T   ||   U    V    W    X    Y    Z'))
        ax.draw_artist(ax.text(5,6,target_text+'\n\n\n\n'+'\n\n\n\n'+user_text))
        ax.draw_artist(ax.text(8,9.5,auto_text))

        # print(auto_text, user_text, refresh_flag.value, next_phrase_flag.value, '*'*10)
        if len(auto_text) != 0 and input_number_num.value != 0 and auto_text != ' ':
            ax.draw_artist(ax.text(8,9.5,auto_text.split()[0][0:int(input_number_num.value)]))
            ax.draw_artist(ax.text(8,9.5,auto_text.split()[0][0:int(input_number_num.value)]))
            ax.draw_artist(ax.text(8,9.5,auto_text.split()[0][0:int(input_number_num.value)]))

        # print("draw >>>", time.time() - start)
        fig.canvas.blit(ax.bbox)

    timer = fig.canvas.new_timer(interval=1)
    timer.add_callback(update)
    timer.start()

    plt.show()

if __name__ ==  '__main__':
    raw_baro = Queue()
    user_text_queue = Queue()
    user_text_queue.put(' ')
    auto_text_queue = Queue()
    auto_text_queue.put(' ')
    target_text_queue = Queue()
    target_text_queue.put('Start from your first gesture!')
    refresh_flag = Value('d',0)
    next_phrase_flag = Value('d',0)
    input_number_flag = Value('d',0)
    input_number_num = Value('d',0)
    input_gesture_id = Value('d',0)
    user_name = input("please input your name: ")
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    data_root_folder = 'user_data/'+user_name+'_'+date
    os.mkdir(data_root_folder)
    os.mkdir(data_root_folder+'/segment_data')
    os.mkdir(data_root_folder+'/segment_data/test')
    P0 = Process(target=process_0_collect_baro,args=(raw_baro,data_root_folder,))
    P1 = Process(target=process_1_gesture_detection,args=(raw_baro,data_root_folder,user_text_queue,refresh_flag,
        next_phrase_flag,auto_text_queue,target_text_queue,input_number_flag,input_number_num,input_gesture_id,))
    P2 = Process(target=process_2_GUI,args=(user_text_queue,refresh_flag,next_phrase_flag,auto_text_queue,
        target_text_queue,input_number_flag,input_number_num,input_gesture_id,))
    P0.start()
    P1.start()
    P2.start()
    P1.join()
    P0.terminate()
    P2.terminate()


