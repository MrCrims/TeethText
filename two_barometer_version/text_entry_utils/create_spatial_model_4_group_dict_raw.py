import json

def dict_to_json_write_file():
    dict = {}
    dict['1_a'] = 1.00; dict['1_b'] = dict['1_a']; dict['1_c'] = dict['1_a']; dict['1_d'] = dict['1_a']; dict['1_e'] = dict['1_a']; dict['1_f'] = dict['1_a']; dict['1_g'] = dict['1_a']; 
    dict['1_h'] = 0.00; dict['1_i'] = dict['1_h']; dict['1_j'] = dict['1_h']; dict['1_k'] = dict['1_h']; dict['1_l'] = dict['1_h']; dict['1_m'] = dict['1_h']; dict['1_n'] = dict['1_h']; 
    dict['1_o'] = 0.00; dict['1_p'] = dict['1_o']; dict['1_q'] = dict['1_o']; dict['1_r'] = dict['1_o']; dict['1_s'] = dict['1_o']; dict['1_t'] = dict['1_o'];  
    dict['1_u'] = 0.00; dict['1_v'] = dict['1_u']; dict['1_w'] = dict['1_u']; dict['1_x'] = dict['1_u']; dict['1_y'] = dict['1_u']; dict['1_z'] = dict['1_u']; 
    ##########################################################################
    dict['2_a'] = 0.00; dict['2_b'] = dict['2_a']; dict['2_c'] = dict['2_a']; dict['2_d'] = dict['2_a']; dict['2_e'] = dict['2_a']; dict['2_f'] = dict['2_a']; dict['2_g'] = dict['2_a']; 
    dict['2_h'] = 1.00; dict['2_i'] = dict['2_h']; dict['2_j'] = dict['2_h']; dict['2_k'] = dict['2_h']; dict['2_l'] = dict['2_h']; dict['2_m'] = dict['2_h']; dict['2_n'] = dict['2_h']; 
    dict['2_o'] = 0.00; dict['2_p'] = dict['2_o']; dict['2_q'] = dict['2_o']; dict['2_r'] = dict['2_o']; dict['2_s'] = dict['2_o']; dict['2_t'] = dict['2_o'];  
    dict['2_u'] = 0.00; dict['2_v'] = dict['2_u']; dict['2_w'] = dict['2_u']; dict['2_x'] = dict['2_u']; dict['2_y'] = dict['2_u']; dict['2_z'] = dict['2_u']; 
    ##########################################################################
    dict['3_a'] = 0.00; dict['3_b'] = dict['3_a']; dict['3_c'] = dict['3_a']; dict['3_d'] = dict['3_a']; dict['3_e'] = dict['3_a']; dict['3_f'] = dict['3_a']; dict['3_g'] = dict['3_a']; 
    dict['3_h'] = 0.00; dict['3_i'] = dict['3_h']; dict['3_j'] = dict['3_h']; dict['3_k'] = dict['3_h']; dict['3_l'] = dict['3_h']; dict['3_m'] = dict['3_h']; dict['3_n'] = dict['3_h']; 
    dict['3_o'] = 1.00; dict['3_p'] = dict['3_o']; dict['3_q'] = dict['3_o']; dict['3_r'] = dict['3_o']; dict['3_s'] = dict['3_o']; dict['3_t'] = dict['3_o'];  
    dict['3_u'] = 0.00; dict['3_v'] = dict['3_u']; dict['3_w'] = dict['3_u']; dict['3_x'] = dict['3_u']; dict['3_y'] = dict['3_u']; dict['3_z'] = dict['3_u']; 
    ##########################################################################
    dict['4_a'] = 0.00; dict['4_b'] = dict['4_a']; dict['4_c'] = dict['4_a']; dict['4_d'] = dict['4_a']; dict['4_e'] = dict['4_a']; dict['4_f'] = dict['4_a']; dict['4_g'] = dict['4_a']; 
    dict['4_h'] = 0.00; dict['4_i'] = dict['4_h']; dict['4_j'] = dict['4_h']; dict['4_k'] = dict['4_h']; dict['4_l'] = dict['4_h']; dict['4_m'] = dict['4_h']; dict['4_n'] = dict['4_h']; 
    dict['4_o'] = 0.00; dict['4_p'] = dict['4_o']; dict['4_q'] = dict['4_o']; dict['4_r'] = dict['4_o']; dict['4_s'] = dict['4_o']; dict['4_t'] = dict['4_o'];  
    dict['4_u'] = 1.00; dict['4_v'] = dict['4_u']; dict['4_w'] = dict['4_u']; dict['4_x'] = dict['4_u']; dict['4_y'] = dict['4_u']; dict['4_z'] = dict['4_u']; 
    ##########################################################################

    print(dict) 
    with open('spatial_model_4_group_dict_raw.json', 'w') as f:
        json.dump(dict, f)  



dict_to_json_write_file()


# def json_file_to_dict():
#     with open('test.json', 'r') as f:
#         dict = json.load(fp=f)
#     print(dict['single_down'])  # {'name': 'many', 'age': 10, 'sex': 'male'}



# json_file_to_dict()
