import numpy as np
# import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import teethtype_module as tm
import json

root_dir = 'user_data/p10_lk_s6_01_10_20_39/'
sample_number = 3

# 1. Visualization Result ------------------------------------------------------------------------------
# train_dir = root_dir + '/segment_data/train'
# # train_dir = root_dir + '/segment_data/test'
# # tm.visual_check_imu_all_random(train_dir)
# tm.visual_check_imu_all(train_dir,sample_number)

# # 2. Teeth Gesture Confusion Matrix --------------------------------------------------------------------
# def confusion_matrix_draw(data_array, label_x_axis, label_y_axis):
#      sn.set(font_scale=1)#for label size
#      df_cm = pd.DataFrame(data_array, label_x_axis, label_y_axis)
#      fig, ax = plt.subplots(figsize = (5,5))
#      ax1 = sn.heatmap(df_cm, annot=True,annot_kws={"size": 15}, cmap="Greens", square=True)# font size
#      label_y = ax1.get_yticklabels()
#      plt.setp(label_y , rotation = 0)
#      label_x = ax1.get_xticklabels()
#      plt.setp(label_x , rotation = 20)

#      fig.tight_layout()   
#      ax.axis('scaled')
#      # plt.title('Heatmap of Flighr Dataset', fontsize = 20) # title with fontsize 20
#      # plt.xlabel('Response', fontsize = 15) # x-axis label with fontsize 15
#      # plt.ylabel('Stimulus', fontsize = 15) # y-axis label with fontsize 15
#      plt.show()

# ground_truth = np.loadtxt(root_dir + '/' + 'ground_truth.txt')
# predict_result = np.loadtxt(root_dir + '/' + 'predict_result.txt')

# count = 0
# if len(ground_truth) == len(predict_result):
#     for i in range(len(ground_truth)):
#         if ground_truth[i] == predict_result[i]:
#             count = count + 1
# else:
#     print('wrong..........................................')

# print('right number: ', count)
# print('accuracy: ', count/len(ground_truth))

# label_x_axis = ['Single_Left_Click','Single_Back_Click','Single_Right_Click',
#                 'Double_Left_Click','Double_Back_Click','Double_Right_Click',
#                 'Left_Slide','Right_Slide']
# label_y_axis = ['Single_Left_Click','Single_Back_Click','Single_Right_Click',
#                 'Double_Left_Click','Double_Back_Click','Double_Right_Click',
#                 'Left_Slide','Right_Slide']

# confusion_matrix_array = confusion_matrix(ground_truth, predict_result)/sample_number
# confusion_matrix_draw(confusion_matrix_array, label_x_axis, label_y_axis)

# 3. Text Entry Speed ---------------------------------------------------------------------------------
time_data = np.loadtxt(root_dir+'user_entry_word_time.txt')
average_time_per_word = np.mean(time_data)
print('WPM: ',60/average_time_per_word)

# 4. Text Entry Accuracy ------------------------------------------------------------------------------
# Uncorrected Error Rate
truth_text = pd.read_csv(root_dir+'truth_text.txt',header=None)
truth_text = np.array(truth_text)
user_text = pd.read_csv(root_dir+'user_entry_word.txt',header=None)
user_text = np.array(user_text)
right_count = 0
for k in range(min(len(truth_text),len(user_text))):
    if truth_text[k] == user_text[k]:
        right_count += 1
uncorrected_error_rate = 1-right_count/min(len(truth_text),len(user_text))
print('Uncorrected Error Rate(UER): ', uncorrected_error_rate)

# Corrected Error Rate
auto_correct_count = 0
with open('word_prob_15000.json', 'r') as f1:
    word_prob_dict = json.load(fp=f1)
with open('spatial_model_4_group_dict.json', 'r') as f2:
    emission_matrix_dict_auto_correct = json.load(fp=f2)
with open('spatial_model_4_group_dict_raw.json', 'r') as f3:
    emission_matrix_dict_raw = json.load(fp=f3)
user_text = pd.read_csv(root_dir+'user_entry_word.txt',header=None)
user_text = np.array(user_text)
user_input_gesture_set = pd.read_csv(root_dir+'user_input_gesture_set.txt',header=None)
user_input_gesture_set = np.array(user_input_gesture_set)
for i in range(len(user_text)):
    predict_top_3_raw = tm.calculate_prob(list(str(user_input_gesture_set[i][0])), word_prob_dict, emission_matrix_dict_raw)
    if user_text[i] not in predict_top_3_raw:
        auto_correct_count += 1
corrected_error_rate = auto_correct_count/len(user_text)
print('Corrected Error Rate(CER): ', corrected_error_rate)

# Total Error Rate
total_error_rate = uncorrected_error_rate + corrected_error_rate
print('Total Error Rate(TER): ', total_error_rate)

# 5. Auto Complete Rate --------------------------------------------------------------------------------
user_text = pd.read_csv(root_dir+'user_entry_word.txt',header=None)
user_text = np.array(user_text)
user_self_input_length = pd.read_csv(root_dir+'user_self_input_word_length.txt',header=None)
user_self_input_length = np.array(user_self_input_length)
auto_complete_rate = np.zeros(len(user_text))
for i in range(len(user_text)):
    each_word_len = len(user_text[i][0])
    auto_complete_rate[i] = 1 - user_self_input_length[i][0] / each_word_len
print('Average Auto-Complete Rate: ',np.mean(auto_complete_rate))





