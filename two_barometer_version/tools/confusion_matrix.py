import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
sn.set(font_scale=1.5)#for label size
# array = [
#      [95,1,1,0,2,0,2,0],
#      [3,90,6,0,1,0,2,0],
#      [0,0,91,2,0,0,2,0],
#      [0,0,0,85,0,0,2,0],
#      [0,0,0,0,75,0,2,0],
#      [0,0,1,4,0,95,2,0],
#      [0,0,1,4,0,3,96,0],
#      [0,0,1,4,0,2,2,99]]        
array = [
     [95,1,1,0,1,0,0,1,0],
     [3,90,6,0,0,2,0,1,0],
     [0,0,91,2,2,1,0,1,0],
     [0,0,0,85,0,1,0,1,0],
     [0,0,0,0,90,1,0,1,0],
     [0,0,0,0,1,95,0,1,0],
     [0,0,0,0,1,0,92,1,0],
     [0,0,0,0,1,2,0,93,0],
     [0,0,0,0,1,3,0,1,96],
     ]    
# df_cm = pd.DataFrame(array, ['M0','M1','M2','M3','M4','M5','M6','M7'], ['M0','M1','M2','M3','M4','M5','M6','M7'])
df_cm = pd.DataFrame(array, ['Single-Left-Click','Double-Left-Click','Triple-Left-Click','Single-Right-Click','Double-Right-Click','Triple-Right-Click', 'Single-Front-Click','Slide-Left','Slide-Right'], ['Single-Left-Click','Double-Left-Click','Triple-Left-Click','Single-Right-Click','Double-Right-Click','Triple-Right-Click', 'Single-Front-Click','Slide-Left','Slide-Right'])

# plt.figure(figsize = (10,7))
fig, ax = plt.subplots(figsize = (5,5))
ax1 = sn.heatmap(df_cm, annot=True,annot_kws={"size": 20}, cmap="Greens", square=True)# font size
label_y = ax1.get_yticklabels()
plt.setp(label_y , rotation = 0)
label_x = ax1.get_xticklabels()
plt.setp(label_x , rotation = 0)

fig.tight_layout()   
ax.axis('scaled')
# plt.title('Heatmap of Flighr Dataset', fontsize = 20) # title with fontsize 20
# plt.xlabel('Response', fontsize = 15) # x-axis label with fontsize 15
# plt.ylabel('Stimulus', fontsize = 15) # y-axis label with fontsize 15
plt.show()