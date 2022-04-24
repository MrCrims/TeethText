import matplotlib as mp 
mp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal,ndimage
from tslearn.preprocessing import TimeSeriesScalerMeanVariance,TimeSeriesScalerMinMax
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from tslearn.metrics import dtw
from sklearn.metrics import accuracy_score
import joblib
from sklearn import preprocessing
import string, codecs, re, json, time, sys, itertools, math, os
# from Bio import trie 
##############################################   useful tools   ###################################################
def read_teeth_list(teeth_list_file_path):
    teeth_list = open(teeth_list_file_path,"r").readlines()
    for i in range(len(teeth_list)):
        teeth = teeth_list[i]
        teeth = teeth.strip('\n')
        teeth = teeth.strip(' ')
        teeth = teeth.strip('\t')
        teeth_list[i] = teeth
    return teeth_list

def generate_id(class_num, sample_per_class):
    '''input: class number and sample per class'''
    order_array = np.zeros(class_num*sample_per_class)
    for m in range(class_num):
        for n in range(sample_per_class):
            order_array[n+sample_per_class*m] = m
    np.random.shuffle(order_array)
    return order_array

def delay_random():
    time_array = np.array([3,4,5])
    np.random.shuffle(time_array)
    time.sleep(time_array[0])

##############################################   visualize data   ###################################################
def visual_check_baro_each(file):
    data = np.loadtxt(file)
    plt.plot(data)
    plt.show()

def visual_check_baro_all(dir, random_state = 0, sample_per_class = 3, norm_flag = 0):
    def take_order(elem):
        return elem.split('_')[-1].split('.')[0]
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    gesture_list = config_para['gesture_list']
    plt.style.use('ggplot')
    plt.figure()
    i = 1
    files = os.listdir(dir)
    for f in files:
        if f.endswith('.DS_Store'):
            os.remove(dir+'/'+f)
    files = os.listdir(dir)
    files.sort(key=take_order)
    file_num = len(files)
    for filename in files:
        if random_state == 0:
            plt.subplot(len(gesture_list),sample_per_class,i)
        else:
            plt.subplot(int(math.sqrt(file_num))+1,int(math.sqrt(file_num))+1,i)
        data = np.loadtxt(dir+'/'+filename)
        # data = preprocessing.MinMaxScaler().fit_transform(data)
        # data = preprocessing.scale(data)
        # data = norm_data(data,1)
        # data = data_extend(data)
        if norm_flag != 0:
            data = norm_data(data,norm_flag)
        plt.plot(data)
        plt.title('Truth'+gesture_list[int(filename.split('_')[-1].split('.')[0])]+'-'+filename.split('_')[0],fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        i += 1
    plt.subplots_adjust(wspace =0.5, hspace =1.5)
    plt.show()

# def visual_check_baro_all_random(dir, norm_flag = 0):
#     plt.style.use('ggplot')
#     plt.figure()
#     i = 1
#     files = os.listdir(dir)
#     for f in files:
#         if f.endswith('.DS_Store'):
#             os.remove(dir+'/'+f)
#     files = os.listdir(dir)
#     print(files)
#     file_num = len(files)
#     for filename in files:
#         # Filter DS.Store file in MAC OS
#         if filename[-4:] == '.txt':
#             plt.subplot(int(math.sqrt(file_num))+1,int(math.sqrt(file_num))+1,i)
#             # print(dir+'/'+filename)
#             data = np.loadtxt(dir+'/'+filename)
#             # data = preprocessing.MinMaxScaler().fit_transform(data)
#             # data = preprocessing.scale(data)
#             # data = norm_data(data,1)
#             # data = data_extend(data)
#             if norm_flag != 0:
#                 data = norm_data(data,norm_flag)
#             plt.plot(data)
#             plt.title(filename,fontsize=6)
#             plt.xticks(fontsize=6)
#             plt.yticks(fontsize=6)
#         i += 1
#     plt.subplots_adjust(wspace =0.5, hspace =1.5)
#     plt.show()
##############################################   segment data   ###################################################
def baro_segment_each(data, theshold_height=0.05, theshold_width=30):#找到比预设值大的气压
    '''input is baro raw data * 6'''
    data_baro_energy = baro_energy(data[:,3:6])
    data_baro_energy_new = np.zeros(len(data_baro_energy))
    # data_baro_energy_seged = np.empty([1,0])
    data_baro_energy_seged = np.zeros((1))
    flag_true = 0
    for k in range(len(data_baro_energy)):
        if data_baro_energy[k] > theshold_height:
            data_baro_energy_new[k] = data_baro_energy[k]
            data_baro_energy_seged = np.append(data_baro_energy_seged, data_baro_energy[k])
            flag_true = k
        elif k - flag_true < theshold_width and k - flag_true > 0 and flag_true > 0:
            data_baro_energy_new[k] = data_baro_energy[k]
            data_baro_energy_seged = np.append(data_baro_energy_seged, data_baro_energy[k])
    return data_baro_energy, data_baro_energy_new, data_baro_energy_seged


def baro_segment_all(raw_dir, seg_dir):
    plt.style.use('ggplot')
    i=1
    for subdir_name in os.listdir(raw_dir):
        for file_name in os.listdir(raw_dir+'/'+subdir_name):
            if file_name[1] == '1':
                baro_file = raw_dir+'/'+subdir_name + '/' + file_name
                baro_data = np.loadtxt(baro_file)
                baro_data_energy, baro_data_energy_new, baro_data_energy_seged = baro_segment_each(baro_data)
                np.savetxt(seg_dir+'/'+subdir_name + '/' + file_name, baro_data_energy_seged)
                plt.subplot(5,10,i)
                i=i+1
                # plt.plot(baro_data_energy)
                # plt.plot(baro_data_energy_new)
                plt.plot(baro_data_energy_seged)
                plt.title(subdir_name,fontsize=8)
                plt.tick_params(labelsize=8)
    plt.show()
##############################################   recognize data   ###################################################
def data_extend(target_data, max_length = 200):
    '''input is 2-d array'''
    target_data_length = len(target_data)
    #extended_data = np.zeros((max_length,1))
    extended_data = np.zeros((max_length,2))
    if target_data_length > max_length:
        #print('too long------------------------------------------------------length: ',target_data_length)
        extended_data = target_data[int((target_data_length-max_length)/2):int((target_data_length-max_length)/2+max_length)]
    else:
        extended_data[int((max_length-target_data_length)/2):int((max_length-target_data_length)/2+target_data_length)] = target_data
    return extended_data

def norm_data(data,normtype,initial_value=0):
    '''1-MinMax for Each, Else-Raw'''
    #二维数据正则化处理
    data_side_1 = data[:,0]
    data_side_2 = data[:,1]
    if normtype == 1:
        # a = data[:,0]
        # b = data[:,1]
        # c = data[:,2]
        # data[:,0] = (a-a.min())/(a.max()-a.min())
        # data[:,1] = (b-b.min())/(b.max()-b.min())
        # data[:,2] = (c-c.min())/(c.max()-c.min())
        #data = (data-data.min())/(data.max()-data.min())
        data_side_1 = (data_side_1-data_side_1.min())/(data_side_1.max()-data_side_1.min())
        data_side_2 = (data_side_2-data_side_2.min())/(data_side_2.max()-data_side_2.min())
        data = np.vstack((data_side_1,data_side_2)).T#这样操作必须保证两个数组大小一样
    elif normtype == 2:
        #data = data - initial_value
        data_side_1 = data_side_1 - initial_value
        data_side_2 = data_side_2 - initial_value
        data = np.vstack((data_side_1, data_side_2)).T
    elif normtype == 0:
        data = data
    return data

# def norm_data(data_3d,normtype):
#     scaler1 = TimeSeriesScalerMeanVariance(mu=0,std=1)
#     scaler2 = TimeSeriesScalerMinMax(min=0,max=1)
#     if normtype == 1:
#         data_3d = scaler1.fit_transform(data_3d)
       
#     elif normtype == 2:
#         data_3d = scaler2.fit_transform(data_3d)
   
#     else:
#         data_3d = data_3d
#     return data_3d


# test normalize (0,1) or (-1,1)
def process_file_train(train_dir,extend_data_length,norm_type=0, initial_value=0):
    train_data = np.empty([0,2])
    train_label = np.empty([0,1])
    files = os.listdir(train_dir)
    for f in files:
        if f.endswith('.DS_Store'):
            os.remove(train_dir+'/'+f)
    files = os.listdir(train_dir)
    for file_name in files:
        filename = train_dir + '/' + file_name
        data_raw = np.loadtxt(filename)
        #二维这里不需要这个reshape,格式在之前的处理已经解决
        #data_raw = data_raw.reshape((len(data_raw),1))#转换为len(data_raw)行，1列
        data_line = data_extend(data_raw,max_length=extend_data_length)
        data_line = norm_data(data_line,norm_type,initial_value)
        #data_line = data_line.reshape((1, len(data_line)))
        train_data = np.append(train_data, data_line, axis=0)
        train_label = np.append(train_label, int(file_name.split('_')[-1].split('.')[0]))

    train_data = train_data.reshape(len(train_label),extend_data_length,2)
    np.save('model/train_data.npy', train_data)#npy文件为numpy专用的二进制文件
    np.save('model/train_label.npy', train_label)

def process_file_test(test_dir,extend_data_length,norm_type=0,initial_value=0):#这个用不上
    test_data = np.empty([0,2])
    test_label = np.empty([0,1])
    files = os.listdir(test_dir)
    for f in files:
        if f.endswith('.DS_Store'):
            os.remove(test_dir+'/'+f)
    files = os.listdir(test_dir)
    for file_name in files:
        filename = test_dir + '/' + file_name
        data_raw = np.loadtxt(filename)
        data_raw = data_raw.reshape((len(data_raw),1))
        data_line = data_extend(data_raw,max_length=extend_data_length)
        data_line = norm_data(data_line,norm_type,initial_value)
        data_line = data_line.reshape((1, len(data_line)))
        test_data = np.append(test_data, data_line, axis=0)
        test_label = np.append(test_label, int(file_name.split('_')[-1].split('.')[0]))
              
    np.save('model/test_data.npy', test_data)
    np.save('model/test_label.npy', test_label)

def train_model(train_data_file, train_label_file, user_name):
    train_X = np.load(train_data_file)
    train_y = np.load(train_label_file)
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn_clf.fit(train_X, train_y)
    print('finished training')
    joblib.dump(knn_clf, 'model/gesture_classify_'+user_name+'.pkl') 

def predict_gesture_visual(test_dir, model_path, random_state = 0, sample_per_class = 3, extend_data_length = 160, norm_type=0, initial_value = 0):
    def take_order(elem):
        return elem.split('_')[-1].split('.')[0]
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    new_clf = joblib.load(model_path) 
    gesture_list = config_para['gesture_list']
    plt.style.use('ggplot')
    plt.figure()
    i = 1
    files = os.listdir(test_dir)
    for f in files:
        if f.endswith('.DS_Store'):
            os.remove(test_dir+'/'+f)
    files = os.listdir(test_dir)
    files.sort(key=take_order)
    file_num = len(files)
    for filename in files:
        if random_state == 0:
            plt.subplot(len(gesture_list),sample_per_class,i)
        else:
            plt.subplot(int(math.sqrt(file_num))+1,int(math.sqrt(file_num))+1,i)
        data = np.loadtxt(test_dir+'/'+filename)
        plot_data = data
        data = data_extend(data.reshape((len(data), 1)),max_length=extend_data_length)
        data = norm_data(data,norm_type,initial_value)
        data = data.reshape((1, len(data)))
        predicted_gesture = new_clf.predict(data)[0]

        
        plot_data = norm_data(plot_data,norm_type,initial_value=initial_value)
        plt.plot(plot_data)
        plt.title('Truth: '+gesture_list[int(filename.split('_')[-1].split('.')[0])]+'\t'+'Pred: '+gesture_list[int(predicted_gesture)]+'-'+filename.split('_')[0],fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        i += 1
    plt.subplots_adjust(wspace =0.5, hspace =1.5)
    plt.show()


def process_file_noise_svm_train_test(head_dir, noise_dir, extend_data_length):
    train_test_data = np.empty([0,extend_data_length])
    train_test_label = np.empty([0,1])
    for file_name in os.listdir(head_dir):
        if file_name[-4:] == '.txt':
            filename = head_dir + '/' + file_name
            data_raw = np.loadtxt(filename)
            data_line = data_extend(data_raw,max_length=extend_data_length)
            data_line_energy = baro_energy(data_line)
            # data_line = norm_data(data_line,1)
            # data_line_energy = data_line_energy.reshape((1, data_line_energy.shape[1], data_line_energy.shape[0]))
            train_test_data = np.append(train_test_data, data_line_energy.reshape((1, len(data_line_energy))), axis=0)
            train_test_label = np.append(train_test_label, int(file_name.split('_')[-1].split('.')[0]))
    
    for file_name in os.listdir(noise_dir):
        if file_name[-4:] == '.txt':
            filename = noise_dir + '/' + file_name
            data_raw = np.loadtxt(filename)
            data_line = data_extend(data_raw,max_length=extend_data_length)
            data_line_energy = baro_energy(data_line)
            # data_line = norm_data(data_line,1)
            # data_line_energy = data_line_energy.reshape((1, data_line_energy.shape[1], data_line_energy.shape[0]))
            train_test_data = np.append(train_test_data, data_line_energy.reshape((1, len(data_line_energy))), axis=0)
            train_test_label = np.append(train_test_label, int(file_name.split('_')[-1].split('.')[0]))
              
    np.save('noise_train_test_data.npy', train_test_data)
    np.save('noise_train_test_label.npy', train_test_label)

############################## Text Auto-complete ####################################

def read_clean_file(file):
    fl = codecs.open( file, "r", "utf-8" )
    txt = fl.read() 
    fl.close()
    cText = txt.lower()
    cText = cText.strip('\n')
    cText = re.sub('[^A-Za-z]+', ' ', cText)#匹配多个连续的非字母并将其换为一个空格
    cText = re.sub('\s{2,}', ' ', cText)
    return cText.split()

def make_word_prob(file):#修改词的频率
    text = read_clean_file(file)
    text_length = len(text)
    word_prob_dict = {}
    for i in range(text_length):
        if text[i] not in word_prob_dict:
            word_prob_dict[text[i]] = 1/text_length
        else:
            word_prob_dict[text[i]] += 1/text_length
    with open('word_prob.json', 'w') as f:
        json.dump(word_prob_dict, f)  
    print('dictionary length: ', len(word_prob_dict))

def process_corpus(dir_name,outfile):
    k = open(outfile, 'a+')
    for parent, dirnames, filenames in os.walk(dir_name):
        for dir_name in filenames:
            txtPath = os.path.join(parent, dir_name) 
            if txtPath[-4:] == '.txt' and txtPath.split('/')[-1][0] != '.':
                print(txtPath)
                f = open(txtPath)
                k.write(f.read()+"\n")
    k.close()
    print ("finished")

# process_corpus('OANC_GrAF','dict_all.txt')

def evaluate_prob(input_num_list, predict_word, word_prob_dict, emission_matrix_dict, alpha = 0.7):
    multi_prob = 1
    n = len(input_num_list)
    m = len(predict_word)
    for k in range(n):
        emission_prob_key = input_num_list[k] + '_' + predict_word[k]
        multi_prob = multi_prob * emission_matrix_dict[emission_prob_key] * math.pow(alpha, (m-n))
    final_prob = multi_prob * word_prob_dict[predict_word]
    final_prob = math.log(final_prob)
    return final_prob

def t9_prediction(word_tree, string, input_num_list, t9_all, word_prob_dict):
    emission_matrix_dict = {'1_q':1/3, '1_w':1/3, '1_e':1/3, '2_r':1/3, '2_t':1/3, '2_y':1/3,
                        '3_u':1/4, '3_i':1/4, '3_o':1/4, '3_p':1/4, '4_a':1/3, '4_s':1/3, '4_d':1/3,
                        '5_f':1/3, '5_g':1/3, '5_h':1/3, '6_j':1/3, '6_k':1/3, '6_l':1/3,
                        '7_z':1/2, '7_x':1/2, '8_c':1/3, '8_v':1/3, '8_b':1/3, '9_n':1/2, '9_m':1/2}
    for predict_word in word_tree.with_prefix(string):
        word_final_prob = evaluate_prob(input_num_list, predict_word, word_prob_dict, emission_matrix_dict)
        # print(predict_word, word_final_prob)
        t9_all[predict_word] = word_final_prob
    return t9_all
        
### make word prob dictionary
# filename = 'dict_all.txt'
# make_word_prob(filename)

### load word prob dictionary
# with open('word_prob.json', 'r') as f:
#     word_prob_dict = json.load(fp=f)


# # load word tree
# word_tree = trie.trie()
# for word in word_prob_dict.keys():
#     word_tree[word] = 1

def number_to_word(input_number_list, word_tree, word_prob_dict):
    # keyboard layout
    # 1 2 3; 4 5 6; 7 8 9
    # QWE RTY UIOP; ASD FGH JKL; ZX CVB NM
    mapping = {1:["q", "w", "e"],
           2:["r", "t", "y"],
           3:["u", "i", "o", "p"],
           4:["a", "s", "d"],
           5:["f", "g", "h"],
           6:["j", "k", "l"],
           7:["z", "x"],
           8:["c", "v", "b"],
           9:["n", "m"]} 
    t9_all = {}
    t1 = time.time()
    digits = map(int, ''.join(input_number_list))
    strings = [''.join(combo) for combo in itertools.product(*(mapping[d] for d in digits))]
    for string in strings:
        t9_prediction(word_tree, string, input_number_list, t9_all, word_prob_dict)

    t9_all_sorted = sorted(t9_all.items(), key=lambda d:d[1], reverse = True)
    t2 = time.time()
    result_top_3 = [candidate_item[0] for candidate_item in t9_all_sorted[0:3]]
    # print('result t9_top_five: ',result_top_5)
    # print('running time: ',t2-t1)
    # if t2-t1>0.1:
    #     print('result t9_top_five: ',result_top_5)
    #     print('running time: ',t2-t1)
    return result_top_3

def word_to_number(word):
    '''input sample: 'you' string type'''
    number_list = []
    mapping = {1:["q", "w", "e"],
           2:["r", "t", "y"],
           3:["u", "i", "o", "p"],
           4:["a", "s", "d"],
           5:["f", "g", "h"],
           6:["j", "k", "l"],
           7:["z", "x"],
           8:["c", "v", "b"],
           9:["n", "m"]} 
    for letter in list(word):
        keys = [x[0] for x in mapping.items() if letter in x[1]]
        number_list.append(str(keys[0]))
    # if len(number_list) > 3:
    #     number_list = number_list[0:3]
    return number_list

def calculate_prob(input_num_list, word_prob_dict, emission_matrix_dict, alpha = 0.7):
    # t1 = time.time()
    predict_word_set = {}
    n = len(input_num_list)
    for potential_word in word_prob_dict.keys():
        if len(potential_word) >= n:
            multi_prob = 1
            m = len(potential_word)
            for k in range(n):
                emission_prob_key = input_num_list[k] + '_' + potential_word[k]
                multi_prob = multi_prob * emission_matrix_dict[emission_prob_key] * math.pow(alpha, (m-n))
            final_prob = multi_prob * word_prob_dict[potential_word]
            predict_word_set[potential_word] = final_prob
    predict_word_set_sorted = sorted(predict_word_set.items(), key=lambda d:d[1], reverse = True)
    result_top_3 = [candidate_item[0] for candidate_item in predict_word_set_sorted[0:3]]
    # t2 = time.time()
    # print(result_top_3)
    # if t2-t1>0.4:
    # print('lasting time: ', t2-t1)
    return result_top_3


# process_file_train('user_data/sg_small_04_29_10_03'+'/segment_data/train',150)
# train_model('train_data.npy','train_label.npy','sg_small')




