import json
import read_baro
import numpy as np

def dict_to_json_write_file(user_name):
    dict = {}
    baro_coefficient = np.array([0.00022,0.0002])
    baro_value = read_baro.read_baro()
    dict['serial_port_pos']='COM3'                                      #串口编号
    dict['initial_baro_value'] = baro_value.tolist()
    dict['threshold_baro_energy'] = (baro_value*baro_coefficient).tolist()
    dict['baro_coefficient'] = baro_coefficient.tolist()
    dict['threshold_baro_length_min'] = 20                              #所能接受的一段气压变化的最少采样点个数
    dict['threshold_baro_length_max'] = 200                             #所能接受的一段气压变化的最多采样点个数
    dict['theshold_width'] = 40                                         #当前一个采样点为超过阈值的有效点时，等待时间
    dict['buffer_size'] = 40
    dict['extend_data_length'] = 200                                    #为保证KNN输入数据维度的统一性的数据格式统一长度
    dict['train_sample_num'] = 3
    dict['test_sample_num'] = 3
    dict['norm_type'] = 1
    dict['model_path'] = "model/gesture_classify_"+user_name+".pkl"
    dict['phrase_path'] = "phrases_set/phrases_set_session_1.txt"
    dict['gesture_list'] = [
        "Single-Left-Click",
        "Double-Left-Click",
        "Single-Down-Click",
        "Double-Down-Click",
        "Triple-Down-Click",
        "Single-Right-Click",
        "Double-Right-Click",
        "Left-Slide",
        "Right-Slide"]
    dict['gesture_max_min_interval_threshold'] = 30
    dict['data_collection_each_gesture_time'] = 5
    print(dict) 
    with open('config_para.json', 'w') as f:
        json.dump(dict, f, indent=4)  



#dict_to_json_write_file('wgj')


# def json_file_to_dict():
#     with open('test.json', 'r') as f:
#         dict = json.load(fp=f)
#     print(dict['single_down'])  # {'name': 'many', 'age': 10, 'sex': 'male'}



# json_file_to_dict()
