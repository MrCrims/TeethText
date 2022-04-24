import serial,time,json
import numpy as np

#这是二维数据版本的（即有两个BMP时）
def read_baro():
    with open('config_para.json', 'r') as f:
        config_para = json.load(fp=f)
    port_position = config_para['serial_port_pos']
    ser = serial.Serial(port_position, baudrate=9600)
    count = 0
    baro_data_side_1 = 0
    baro_data_side_2 = 0
    baro_value_side_1 = 0
    baro_value_side_2 = 0
    baro_value = np.empty([0,2])
    time1 = time.time()
    while time.time() - time1 < 4:
        if 2 < time.time() - time1 < 3:
            try:
                baro_data = ser.readline().decode("utf-8")
                baro_data_side_1 = float(baro_data.split(',')[0])
                baro_data_side_2 = float(baro_data.split(',')[1].split('\r')[0])
                baro_value_side_1 += baro_data_side_1
                baro_value_side_2 += baro_data_side_2
                if baro_data_side_1 > 150000 or baro_data_side_1 < 50000 or baro_data_side_2 > 150000 or baro_data_side_2 < 50000:
                    continue
                #baro_value += baro_data
                count += 1
            except:
                continue
    baro_value = np.vstack((baro_value_side_1,baro_value_side_2)).T/count
    return baro_value

if __name__ == '__main__':
    while True:
        baro = read_baro()
        print(baro)