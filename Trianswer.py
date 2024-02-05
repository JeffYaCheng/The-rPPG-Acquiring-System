import os
import asyncio
import sys

from bleak import BleakScanner, BleakClient
import numpy as np
import warnings

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from datetime import datetime, timedelta
import json
import math
from tkinter import messagebox

import cv2
import time
from tkinter import *
import PIL
from PIL import ImageTk
from PIL import Image

DATA_VAL_UUID = "0000a001-0000-1000-8000-00805f9b34fb"
DEVICE_NAME_FILTER = "Tri"
Device_SN_flag = True
Device_list = []
Device_list_advData = []
SN_str = ""
Target_device = None
conn_connected = False
FLAG_ALLOW_CONN = True
FLAG_SENSING_EN = False
client = None
freq = 1000
pkg_NO_pre = 0
warnings.filterwarnings('ignore')
FLAG_APP_CLOSE = False
FLAG_DEV_SELECT = False
FLAG_WINDOW_CLOSE = False
CH_NUM = 3
PPG_VALID_LEN_INDEX = 72
frm_Ch = [None, None, None]

# record video
camera = None
video = None
image_width = int(640)
image_height = int(480)
camera_num = int(1)
# File related
TriAnswer_PROG_ROOT = os.path.abspath(os.getcwd())
TriAnswer_Record_Dir = os.path.join(TriAnswer_PROG_ROOT, 'TriAnswer_Records')
TA_FileHandle = [None, None, None, None]
CH_SpeedText = ['Fast_1KHz', 'Medium_500Hz', 'Slow_333Hz', 'Default_100Hz']
Btn_Record_State = 0  # 0: Rec, 1:Stop

'''
FLOW_CTRL_STATE : 
0 : init                -> unlock MQTT, make all reset, After select the device, turn Scan btn into Connect btn
1 : Click Connect btn   -> Lock MQTT
2 : when Connected   -> turn on the text blinking effect, turn Connect btn into Stop btn
'''
FLOW_CTRL_STATE = 0


# TK new UI
Log_text_val = ""
Btn_BLE_control_color = ["#5CADAD", "#02C874", "#FF2D2D"]
# 36 : TriAnswer packet size / CH
waveform_buffer_len = math.ceil((freq * 7) / 36) * 36
Signal_FRAME_LEN = waveform_buffer_len / freq
waveform_buffer = np.zeros((CH_NUM+1, waveform_buffer_len))
waveform_buffer[waveform_buffer == 0] = np.nan
waveform_buffer_index = np.zeros(CH_NUM+1, int)
waveform_draw_period = 500
PPG_DC_GET_FLAG = False
PPG_DC = [0, 0]

# when window closing


def on_WinClosing():
    global FLAG_WINDOW_CLOSE, FLAG_SENSING_EN, FLOW_CTRL_STATE
    if conn_connected:
        # messagebox.showwarning("Warning", "Please click the [Stop] button before leaving the program")
        FLAG_SENSING_EN = False
        FLAG_WINDOW_CLOSE = True
        FLOW_CTRL_STATE = 0
    else:
        window.destroy()


# FLOW_CTRL_STATE BTN action
def FLOW_CTRL_BTN_action():
    global client, FLAG_ALLOW_CONN, FLAG_SENSING_EN, MQTTClient, MQTT_ACT, MQTT_PWD, MQTT_IP, MQTT_PORT, MQTTMsg_Level, \
        async_loop, FLOW_CTRL_STATE

    if FLOW_CTRL_STATE == 0:  # Before device selecting
        print('> Start scanning ...')
        log_update('> Start scanning ...')
        threading.Thread(target=_scan_thread, args=(async_loop,)).start()
    if FLOW_CTRL_STATE == 1:  # Try to connect to device
        log_update('> Start connecting ...')

        # Disable the setting views
        Combobox_BLE_Device['state'] = 'disable'

        client = BleakClient(Target_device.address)
        FLAG_ALLOW_CONN = True
        FLAG_SENSING_EN = True
        print(f'> Start connecting {Target_device.address} ...')
        threading.Thread(target=_connect_to_device_thread,
                         args=(async_loop,)).start()
        threading.Thread(target=turn_on_camera).start()
        threading.Thread(target=waveform_draw_task).start()  # draw waveform
    if FLOW_CTRL_STATE == 2:  # Try to disconnect to device
        FLAG_SENSING_EN = False
        FLOW_CTRL_STATE = 0


def device_selected(event):
    global Target_device, FLOW_CTRL_STATE
    FLOW_CTRL_STATE = 1
    Target_device = Device_list[Combobox_BLE_Device.current()]
    Btn_BLE_control['text'] = 'Connect to Device'
    Btn_BLE_control['bg'] = Btn_BLE_control_color[FLOW_CTRL_STATE]


def _scan_thread(async_loop):
    async_loop.run_until_complete(scan())


# To discover BLE devices nearby
async def scan():
    global Device_list, Device_list_advData
    Device_list = []
    Device_list_advData = []

    scanner = BleakScanner()
    scanner.register_detection_callback(detection_callback)
    await scanner.start()
    await asyncio.sleep(3)
    await scanner.stop()

    # Device Filter
    dev_list_str = []
    if len(Device_list) > 0:
        for ind in range(len(Device_list)):
            dev_list_str.append(
                f'{Device_list_advData[ind].local_name}({Device_list[ind].rssi})')
        log_update(
            "> Select the device you want to connect at the [drop-down] list")
        Combobox_BLE_Device['state'] = 'readonly'
        Combobox_BLE_Device['value'] = dev_list_str
    else:
        log_update("> No TriAnswer Device Found !")


def detection_callback(device, advertisement_data):
    global Target_device
    if device is not None:
        if DEVICE_NAME_FILTER in f'{device}':
            allow_append = True
            for ind in range(len(Device_list)):
                if Device_list[ind].address == device.address:
                    allow_append = False
            if allow_append:
                Device_list.append(device)
                Device_list_advData.append(advertisement_data)


async def handle_rx_data(_: int, data: bytearray):
    global pkg_NO_pre, Device_SN_flag, SN_str, waveform_buffer_index, waveform_buffer, PPG_DC_GET_FLAG, PPG_DC

    # BLE Data to CH 1 & 2 ECG Values
    ECG_temp_arr = np.zeros(PPG_VALID_LEN_INDEX)
    for ind in range(0, PPG_VALID_LEN_INDEX):
        ECG_temp_arr[ind] = (data[ind] & 0xFF) * 3600 / (2 ** 8)  # Unit : mV

    # BLE Data to PPG Values
    PPG_DATA_INDEX_BASE = PPG_VALID_LEN_INDEX + 1
    PPG_VALID_LEN = round((data[PPG_VALID_LEN_INDEX] & 0xFF) / 4)
    PPG_temp_arr = np.zeros((2, PPG_VALID_LEN))
    for ind in range(0, PPG_VALID_LEN):
        PPG_temp_arr[0][ind] = (data[PPG_DATA_INDEX_BASE + ind * 4] & 0xFF) + (
            data[PPG_DATA_INDEX_BASE + ind * 4 + 1] & 0xFF) * 256
        PPG_temp_arr[1][ind] = (data[PPG_DATA_INDEX_BASE + ind * 4 + 2] & 0xFF) + (
            data[PPG_DATA_INDEX_BASE + ind * 4 + 3] & 0xFF) * 256

    if not PPG_DC_GET_FLAG:
        PPG_DC = [PPG_temp_arr[0][0], PPG_temp_arr[1][0]]
        PPG_temp_arr = [PPG_temp_arr[0][1:], PPG_temp_arr[1][1:]]
        PPG_DC_GET_FLAG = True

    # Waveform Processing (the right the newer)
    wave_temp_array = [[], [], [], []]

    for ch_ind in range(0, 4):
        # Prepare the drawing data per channel (先全部填入，繪圖前才做降頻)
        if ch_ind < 2:  # CH1~2
            wave_temp_array[ch_ind] = ECG_temp_arr[ch_ind::2]
        else:  # CH3~4 : 10倍頻至1000Hz再降頻
            wave_temp_array[ch_ind] = np.repeat(
                PPG_temp_arr[ch_ind - 2], CH_SpeedMode[ch_ind])-PPG_DC[ch_ind - 2]

        # Write to file
        if Btn_Record_State == 1:  # when record bottom trigger and the function change the state it will start write  txt file
            # start_recording_2 = time.time() ## test thread
            # print("start_save_file",start_recording_2) 
            if ch_ind!=1:
                for ele in wave_temp_array[ch_ind][::CH_SpeedMode[ch_ind]]:
                    TA_FileHandle[ch_ind].write(f'{ele}\n')

        if (waveform_buffer_index[ch_ind] + len(wave_temp_array[ch_ind])) <= len(
                waveform_buffer[ch_ind]):  # buffer not full
            waveform_buffer[ch_ind][
                waveform_buffer_index[ch_ind]:(waveform_buffer_index[ch_ind] + len(wave_temp_array[ch_ind]))] = \
                wave_temp_array[ch_ind]
            waveform_buffer_index[ch_ind] = waveform_buffer_index[ch_ind] + \
                len(wave_temp_array[ch_ind])
        else:
            waveform_buffer[ch_ind][:-len(wave_temp_array[ch_ind])] = waveform_buffer[ch_ind][
                len(wave_temp_array[ch_ind]):]
            waveform_buffer[ch_ind][-len(wave_temp_array[ch_ind]):] = wave_temp_array[ch_ind]


def _connect_to_device_thread(async_loop):
    async_loop.run_until_complete(connect_to_device())


async def connect_to_device():
    global conn_connected, client, Target_device, MQTTClient, FLOW_CTRL_STATE, \
        waveform_buffer, waveform_buffer_index, PPG_DC_GET_FLAG

    while FLAG_SENSING_EN:
        if FLAG_ALLOW_CONN:
            try:
                await client.connect()
                await client.get_services()
                conn_connected = client.is_connected()
                print("> Device connected !")

                if conn_connected:
                    FLOW_CTRL_STATE = 2
                    Btn_Record['state'] = 'normal'
                    Btn_Record['bg'] = "#ef5350"
                    Btn_BLE_control['text'] = 'Stop Connection'
                    Btn_BLE_control['bg'] = Btn_BLE_control_color[FLOW_CTRL_STATE]
                    log_update(
                        "> Connect to TriAnswer and try to initiate the measurement ...")
                    # waveform_draw_task()

                    await client.start_notify(DATA_VAL_UUID, handle_rx_data)
                    log_update('> Sensing ...')

                    while FLAG_SENSING_EN:
                        if not conn_connected:
                            print("> Not conn_connected")
                            break
                        await asyncio.sleep(0.3)
                else:
                    print(f"> Failed to connect to Device")
            except Exception as e:
                print(e)
        else:
            await asyncio.sleep(1.0)

    # After FLAG_SENSING_EN flag turns to False -> stop_notify and disconnect BLE device
    log_update(f'> [BLE] Try to disconnect...')
    await client.disconnect()
    closeRecordFile()  # i think if disconnect and forget stop record will use it to close file

    if not FLAG_WINDOW_CLOSE:
        conn_connected = False
        client = None
        log_update(f'> [BLE] {Target_device.address} disconnected...')
        Target_device = None
        MQTTClient = None
        window.after(300, None)
        # log_update(f'> Procedure is Stopped ...')
        # window.after(100, None)
        FLOW_CTRL_STATE = 0
        log_update(
            f'> Click on the [Scan] button to connect to TriAnswer Device...')
        Btn_BLE_control['text'] = 'Scan'
        Btn_BLE_control['bg'] = Btn_BLE_control_color[FLOW_CTRL_STATE]
        Combobox_BLE_Device['value'] = []
        Combobox_BLE_Device.set('')
        Combobox_BLE_Device['state'] = 'disable'
        Btn_Record['state'] = 'disable'
        Btn_Record['bg'] = 'grey'
        PPG_DC_GET_FLAG = False

        # Drawing param reset
        reset_all_figure()

    else:
        window.quit()


def reset_all_figure():
    global waveform_buffer, waveform_buffer_index
    waveform_buffer = np.zeros((CH_NUM+1, waveform_buffer_len))
    waveform_buffer[waveform_buffer == 0] = np.nan
    waveform_buffer_index = np.zeros(CH_NUM+1, int)

    for ch_num in range(0, 2):
        mat_ax[ch_num].clear()
        number = ch_num
        if (ch_num == 1):
            number = 2
        mat_ax[ch_num].plot(time_array[::CH_SpeedMode[number]],
                            waveform_buffer[number][::CH_SpeedMode[number]],
                            color='r', linewidth=1, alpha=0.7)

        # axis setting
        mat_ax[ch_num].set_xlim(0, Signal_FRAME_LEN)
        mat_ax[ch_num].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
        mat_ax[ch_num].set_xticklabels([])
        mat_ax[ch_num].grid(axis='x')
        mat_ax[ch_num].yaxis.set_visible(False)
        mat_ax[ch_num].set_yticks([])

    mat_ax[0].set_title('ECG')
    mat_ax[1].set_title('PPG')
    canvas.draw()


def reset_figure(ch_num):
    global waveform_buffer, waveform_buffer_index
    waveform_buffer[ch_num] = np.zeros(waveform_buffer_len)
    waveform_buffer[ch_num][waveform_buffer[ch_num] == 0] = np.nan
    waveform_buffer_index[ch_num] = 0

    mat_ax[ch_num].clear()
    mat_ax[ch_num].plot(time_array[::CH_SpeedMode[ch_num]],
                        waveform_buffer[ch_num][::CH_SpeedMode[ch_num]],
                        color='r', linewidth=1, alpha=0.7)

    # axis setting
    mat_ax[ch_num].set_xlim(0, Signal_FRAME_LEN)
    mat_ax[ch_num].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
    mat_ax[ch_num].set_xticklabels([])
    mat_ax[ch_num].grid(axis='x')
    mat_ax[ch_num].yaxis.set_visible(False)
    mat_ax[ch_num].set_yticks([])
    canvas.draw()


def log_blinking_show():
    Log_text['text'] = Log_text_val


def log_update(input_str):
    global Log_text_val
    Log_text_val = input_str
    Log_text['text'] = ''
    window.after(50, log_blinking_show)


def waveform_draw_task():
    global conn_connected, client, Target_device, MQTTClient, FLOW_CTRL_STATE, \
        waveform_buffer, waveform_buffer_index, PPG_DC_GET_FLAG, FLAG_SENSING_EN
    while (FLAG_SENSING_EN):
        for ch_num in range(0, 2):
            mat_ax[ch_num].clear()

            if ch_num < 1: #delete ch2
                mat_ax[ch_num].plot(time_array[::CH_SpeedMode[ch_num]], #CH0
                                    waveform_buffer[ch_num][::CH_SpeedMode[ch_num]],
                                    color='r', linewidth=1, alpha=0.7)
            else:
                # CH3
                # 紅外光
                '''
                mat_ax[ch_num].plot(time_array[::CH_SpeedMode[ch_num+1]],#1+1=CH2
                                    waveform_buffer[ch_num +
                                                    1][::CH_SpeedMode[ch_num+1]],
                                    color='r', linewidth=1, alpha=0.7)
                '''
                # CH4
                # 紅光
                mat_ax[ch_num].plot(time_array[::CH_SpeedMode[ch_num + 2]], #CH3
                                    waveform_buffer[ch_num +
                                                    2][::CH_SpeedMode[ch_num + 2]],
                                    color='b', linewidth=1, alpha=0.7)
            # axis setting
            mat_ax[ch_num].set_xlim(0, Signal_FRAME_LEN)
            mat_ax[ch_num].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
            mat_ax[ch_num].set_xticklabels([])
            mat_ax[ch_num].grid(axis='x')
            mat_ax[ch_num].yaxis.set_visible(False)
            mat_ax[ch_num].set_yticks([])
        mat_ax[0].set_title('ECG')
        mat_ax[1].set_title('PPG')
        # update figure
        canvas.draw()
        time.sleep(0.5)

    # if FLOW_CTRL_STATE == 2:  # when connected
    #    window.after(waveform_draw_period, waveform_draw_task)


class ipcamCapture:
    def __init__(self):
        global camera_num
        self.Frame = []
        self.status = False
        self.isstop = False
        self.connect = False
        # 攝影機連接 # 1 外接
        self.capture = cv2.VideoCapture(camera_num)
        if not self.capture.isOpened():
            print('just have one camera')
            self.capture = cv2.VideoCapture(0)

        # 關掉相機自動的功能
        exp = self.capture.get(cv2.CAP_PROP_EXPOSURE)
        wb = self.capture.get(cv2.CAP_PROP_WB_TEMPERATURE)
        fc = self.capture.get(cv2.CAP_PROP_FOCUS)
        self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)  # 關掉自動白平衡 0
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.capture.set(cv2.CAP_PROP_EXPOSURE, exp)
        self.capture.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        self.capture.set(cv2.CAP_PROP_FOCUS, fc)

        if not self.capture.isOpened():
            print("Cannot open camera")
            exit()
        else:
            print("turn on camera")
            self.connect = True

    def start(self):
        # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        # 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
        self.capture.release()

    def getframe(self):
        # 當有需要影像時，再回傳最新的影像。
        return self.Frame.copy()

    def queryframe(self):
        global video
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
            if not self.status:
                print("Can't receive frame (stream end?). Exiting ...")
            else:
                # 記錄影片
                while (Btn_Record_State):
                    self.status, self.Frame = self.capture.read()
                    video.write(self.Frame)
        self.capture.release()


def turn_on_camera():
    print('enter function turn_on_camera')
    global camera, video, FLAG_SENSING_EN, Btn_Record_State
    ipcam = ipcamCapture()
    ipcam.start()
    # 讓相機初始設定
    while (not ipcam.connect):
        time.sleep(1)

    # 只要是sensing 時，照相機也會一直開著
    while (FLAG_SENSING_EN):
        # 擷取影像
        frame = ipcam.getframe()
        if ipcam.status:
            try:
                frame = cv2.resize(frame, (image_width, image_height),
                                   interpolation=cv2.INTER_AREA)  # 有可能第一張收不到frame
            except:
                print('can not get frame')
                continue
            # resize image to 640*480
            frame = cv2.flip(frame, 1)    # 左右翻轉
            # add rectangle
            cv2.rectangle(frame, (int(5*frame.shape[1]/10+frame.shape[1]/20), int(
                frame.shape[0]/5)), (int(9*frame.shape[1]/10), int(4*frame.shape[0]/5)), (0, 0, 255), 2)
            cv2.putText(frame,'Hand',(int(5*frame.shape[1]/10+frame.shape[1]/20), int(frame.shape[0]/5)-5),cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (int(1*frame.shape[1]/10), int(1*frame.shape[0]/6)), (int(
                5*frame.shape[1]/10), int(5*frame.shape[0]/6)), (0, 0, 255), 2)
            cv2.putText(frame,'Face',(int(1*frame.shape[1]/10), int(1*frame.shape[0]/6)-5),cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 0, 255), 1, cv2.LINE_AA)
            # show frame on tk
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(frame)
            # ilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=pilImage)
            movieLabel.config(image=imgtk)
            movieLabel.image = imgtk
            # window.update()  # update 照片

            start_recording = time.time()
            while (Btn_Record_State):
                now_time = time.time()
                frame = ipcam.getframe()
                frame = cv2.resize(
                    frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
                frame = cv2.flip(frame, 1)    # 左右翻轉
                # 放時間
                pos = (450, 10)
                x, y = pos
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 2
                font_thickness = 1
                text = f'time{str(int(now_time-start_recording))}s'
                text_size, _ = cv2.getTextSize(
                    text, font, font_scale, font_thickness)
                text_w, text_h = text_size
                cv2.rectangle(
                    frame, pos, (x + text_w, y + text_h), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, y + text_h + font_scale - 1),
                            font, font_scale, (255, 255, 255), font_thickness)
                # 放框框
                cv2.rectangle(frame, (int(5*frame.shape[1]/10+frame.shape[1]/20), int(
                frame.shape[0]/5)), (int(9*frame.shape[1]/10), int(4*frame.shape[0]/5)), (0, 0, 255), 2)
                cv2.putText(frame,'Hand',(int(5*frame.shape[1]/10+frame.shape[1]/20), int(frame.shape[0]/5)-5),cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (int(1*frame.shape[1]/10), int(1*frame.shape[0]/6)), (int(
                5*frame.shape[1]/10), int(5*frame.shape[0]/6)), (0, 0, 255), 2)
                cv2.putText(frame,'Face',(int(1*frame.shape[1]/10), int(1*frame.shape[0]/6)-5),cv2.FONT_HERSHEY_SIMPLEX,
  1, (0, 0, 255), 1, cv2.LINE_AA)
                # show frame on tk
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pilImage = Image.fromarray(frame)
                # pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
                imgtk = ImageTk.PhotoImage(image=pilImage)
                movieLabel.config(image=imgtk)
                movieLabel.image = imgtk
            out_time = time.time()
    # 釋放該攝影機裝置
    print('turn off camera')
    ipcam.stop()
    print('leave function turn_on_camera')
    # 放logo
    frame = cv2.imread("./logo.jpg")
    frame = cv2.resize(frame, (image_width, image_height),
                       interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(frame)
    # ilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=pilImage)
    movieLabel.config(image=imgtk)
    movieLabel.image = imgtk


def createRecordFile():
    global TA_FileHandle, video  # add global variable video

    dirIsExist = os.path.exists(TriAnswer_Record_Dir)
    if not dirIsExist:
        os.makedirs(TriAnswer_Record_Dir)

    # Disable the setting frame
    """
    chk_CH1.configure(state='disable')
    chk_CH2.configure(state='disable')
    chk_CH3.configure(state='disable')
    for ch_ind in range(0, 3):
        for childView in frm_Ch[ch_ind].winfo_children():
            childView.configure(state='disable')
    """

    now_time = datetime.now()
    time_str = now_time.strftime('%Y_%m_%d_%H_%M_%S')
    file_dir = r = f'{TriAnswer_Record_Dir}/' + time_str
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    for ch_num in range(0, 3):
        if ch_num == 0:  # CH1
            FileName = f'ECG.csv'
            TA_FileHandle[ch_num] = open(os.path.join(file_dir, FileName), "w")
        elif ch_num == 1:  # CH2
            FileName = f'CH{ch_num + 1}.csv'
            #不存ch2
            #TA_FileHandle[ch_num] = open(os.path.join(file_dir, FileName), "w")
        else:
            # CH3
            FileName = f'PPG_IR.csv'
            TA_FileHandle[ch_num] = open(os.path.join(file_dir, FileName), "w")
            # CH4
            FileName = f'PPG_R.csv'
            TA_FileHandle[ch_num +
                          1] = open(os.path.join(file_dir, FileName), "w")

    # add video file
    video_name = file_dir + r'/video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 30, (640, 480))


def closeRecordFile():
    global TA_FileHandle, Btn_Record_State

    if TA_FileHandle[0] is not None or TA_FileHandle[1] is not None or TA_FileHandle[2] is not None or TA_FileHandle[
            3] is not None:
        for ch_num in range(0, 4):
            if ch_num!=1:
                TA_FileHandle[ch_num].close()

        TA_FileHandle = [None, None, None, None]

    # Enable the setting frame
    Btn_Record['text'] = 'Start\nREC'
    Btn_Record['bg'] = "#ef5350"
    Btn_Record_State = 0
    """
    chk_CH1.configure(state='normal')
    chk_CH2.configure(state='normal')
    chk_CH3.configure(state='normal')
    for ch_ind in range(0, 3):
        if CH_EN[ch_ind].get():
            for childView in frm_Ch[ch_ind].winfo_children():
                childView.configure(state='normal')
    """


"""
def CH_SPEED_EN(ch_num):
    if CH_EN[ch_num].get():
        for childView in frm_Ch[ch_num].winfo_children():
            childView.configure(state='normal')
    else:
        for childView in frm_Ch[ch_num].winfo_children():
            childView.configure(state='disable')
"""


def Btn_Record_Ctrl():
    global Btn_Record_State
    if Btn_Record_State == 0:
        createRecordFile()  # just creat file
        Btn_Record['text'] = 'STOP\nREC'
        Btn_Record['bg'] = "#96d1b1"
        start_recording = time.time()
        Btn_Record_State = 1  # set to one start write data fo file
        print("set_record_state=1 time=", start_recording)
    elif Btn_Record_State == 1:
        closeRecordFile()


if __name__ == "__main__":

    # Initiate TK GUI
    window = tk.Tk()
    window.title("Yutech - TriAnswer Controller")
    window.geometry('1600x850')  # 注意是string，而不是數字 #inital window size
    window.resizable(width=False, height=False)
    font = ('Arial', 15, 'bold')  # 字體大小
    figsize = (9, 6)  # 繪圖大小

    frm_BLE = tk.LabelFrame(
        master=window, text="TriAnswer Setting & Control", padx=5, pady=5)
    CH_SpeedMode = [int(1), int(1), int(10), int(10)] #::等間距取值
    """
    # BLE block
    frm_BLE = tk.LabelFrame(master=window, text="TriAnswer Setting & Control", padx=5, pady=5)

    # CH Enable block
    frm_ChEn = tk.LabelFrame(master=frm_BLE, text="Channel Settings", padx=5, pady=5)

    #CH_EN = [tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()]
    CH_SpeedMode = [int(1), int(1), int(10), int(10)]

    # CH1
    chk_CH1 = tk.Checkbutton(master=frm_ChEn, text="啟用CH1", variable=CH_EN[0], command=lambda: CH_SPEED_EN(0))
    chk_CH1.grid(row=0, column=0, padx=1, pady=5)

    frm_Ch[0] = tk.LabelFrame(master=frm_ChEn, text="", bd=0, padx=5, pady=0)
    tk.Radiobutton(master=frm_Ch[0], text='Fast', variable=CH_SpeedMode[0], value=1,
                   command=lambda: reset_figure(0)).grid(row=0, column=0, padx=1, pady=5)
    tk.Radiobutton(master=frm_Ch[0], text='Medium', variable=CH_SpeedMode[0], value=2,
                   command=lambda: reset_figure(0)).grid(row=0, column=1, padx=1, pady=5)
    tk.Radiobutton(master=frm_Ch[0], text='Slow', variable=CH_SpeedMode[0], value=3,
                   command=lambda: reset_figure(0)).grid(row=0, column=2, padx=1, pady=5)
    frm_Ch[0].grid(row=0, column=1, padx=5, pady=0, sticky='ew')
    CH_EN[0].set(1)
    CH_SpeedMode[0].set(1)

    # CH2
    chk_CH2 = tk.Checkbutton(master=frm_ChEn, text="啟用CH2", variable=CH_EN[1], command=lambda: CH_SPEED_EN(1))
    chk_CH2.grid(row=1, column=0, padx=1, pady=5)

    frm_Ch[1] = tk.LabelFrame(master=frm_ChEn, text="", bd=0, padx=5, pady=0)
    tk.Radiobutton(master=frm_Ch[1], text='Fast', variable=CH_SpeedMode[1], value=1,
                   command=lambda: reset_figure(1)).grid(row=0, column=0, padx=1, pady=5)
    tk.Radiobutton(master=frm_Ch[1], text='Medium', variable=CH_SpeedMode[1], value=2,
                   command=lambda: reset_figure(1)).grid(row=0, column=1, padx=1, pady=5)
    tk.Radiobutton(master=frm_Ch[1], text='Slow', variable=CH_SpeedMode[1], value=3,
                   command=lambda: reset_figure(1)).grid(row=0, column=2, padx=1, pady=5)
    frm_Ch[1].grid(row=1, column=1, padx=5, pady=0, sticky='ew')
    CH_EN[1].set(1)
    CH_SpeedMode[1].set(1)

    # CH3
    chk_CH3 = tk.Checkbutton(master=frm_ChEn, text="啟用CH3", variable=CH_EN[2], command=lambda: CH_SPEED_EN(2))
    chk_CH3.grid(row=2, column=0, padx=1, pady=5)

    frm_Ch[2] = tk.LabelFrame(master=frm_ChEn, text="", bd=0, padx=5, pady=0)
    tk.Radiobutton(master=frm_Ch[2], text='Default', variable=CH_SpeedMode[2], value=10,
                   command=lambda: reset_figure(2)).grid(row=0, column=0, padx=1, pady=5)
    frm_Ch[2].grid(row=2, column=1, padx=5, pady=0, sticky='ew')
    CH_EN[2].set(1)
    CH_SpeedMode[2].set(10)

    # Make CH4 the same settings as CH3
    CH_EN[3].set(1)
    CH_SpeedMode[3].set(10)


    frm_ChEn.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
    """
    # BLE Connection block
    frm_BleCon = tk.LabelFrame(
        master=frm_BLE, text="BLE Connection", padx=5, pady=5)

    # BLE Device
    Combobox_BLE_Device = ttk.Combobox(
        master=frm_BleCon, width=28, state='disable')
    Combobox_BLE_Device.bind('<<ComboboxSelected>>', device_selected)
    Combobox_BLE_Device.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

    # BLE Control BTN
    Btn_BLE_control = tk.Button(master=frm_BleCon, text="Scan", width=10, font=font,
                                bg=Btn_BLE_control_color[FLOW_CTRL_STATE], fg="yellow", command=FLOW_CTRL_BTN_action)
    Btn_BLE_control.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
    # Record BTN
    Btn_Record = tk.Button(master=frm_BleCon, text='Start\nREC', width=6, state='disable', font=font,
                           bg="grey", fg="white", command=Btn_Record_Ctrl)
    Btn_Record.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky='nsew')
    Btn_Record_State = 0

    frm_BleCon.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

    frm_BLE.grid(row=0, column=1, padx=5, sticky='nsew')

    # add movie
    # canvas_v = tk.Canvas(master=frm_BLE, width=image_width, height=image_height)  # 加入 Canvas 畫布
    # canvas_v.place(x=0,y=0)
    # canvas.grid()
    movieLabel = tk.Label(master=frm_BLE)
    movieLabel.grid()
    # 先放一張logo
    frame = cv2.imread("./logo.jpg")
    frame = cv2.resize(frame, (image_width, image_height),
                       interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(frame)
    # ilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=pilImage)
    movieLabel.config(image=imgtk)
    movieLabel.image = imgtk
    # Figure block
    number_of_figure = 2
    frm_fig = tk.LabelFrame(
        master=window, text="Waveform Monitor", padx=5, pady=5)
    time_array = np.arange(waveform_buffer_len) / freq
    mat_fig, mat_ax = plt.subplots(
        number_of_figure, 1, figsize=figsize, dpi=100, constrained_layout=True)

    mat_fig.patch.set_facecolor('#F0F0F0')
    mat_fig.patch.set_alpha(1.0)

    for i in range(0, number_of_figure):
        mat_ax[i].set_facecolor('white')
        mat_ax[i].set_xlim(0, Signal_FRAME_LEN)
        mat_ax[i].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
        mat_ax[i].set_xticklabels([])
        mat_ax[i].grid(axis='x')
        mat_ax[i].yaxis.set_visible(False)
        mat_ax[i].set_yticks([])
    
    mat_ax[0].set_title('ECG')
    mat_ax[1].set_title('PPG')
    canvas = FigureCanvasTkAgg(mat_fig, master=frm_fig)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    frm_fig.grid(row=0, column=0, padx=5, sticky='nsew')

    # Log block
    frm_log = tk.LabelFrame(master=window, text="Log", padx=5, pady=5)
    Log_text = tk.Label(master=frm_log, bg='#D1E9E9', anchor="w")
    Log_text['text'] = f'> Click on the [Scan] button to connect to TriAnswer Device...'
    Log_text.pack(fill='x')
    frm_log.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

    # Asyncio
    async_loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    window.protocol("WM_DELETE_WINDOW", on_WinClosing)
    window.mainloop()
