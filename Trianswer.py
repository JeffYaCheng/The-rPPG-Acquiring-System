import os
import shutil
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

import glob
import pandas as pd
from scipy.signal import firwin, lfilter, filtfilt
from math import ceil
import pandas as pd

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
warnings.filterwarnings("ignore")
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
exposure = int(-5)
gain = int(1)
imgpath = None
# File related
TriAnswer_PROG_ROOT = os.path.abspath(os.getcwd())
TriAnswer_Record_Dir = os.path.join(TriAnswer_PROG_ROOT, "TriAnswer_Records")
TA_FileHandle = [None, None, None, None]
CH_SpeedText = ["Fast_1KHz", "Medium_500Hz", "Slow_333Hz", "Default_100Hz"]
Btn_Record_State = 0  # 0: Rec, 1:Stop
save_file_dir = None  # 存下來的檔案名稱 2024_02_05_10_59_..
save_time_complete = False
ipcam = None
"""
FLOW_CTRL_STATE : 
0 : init                -> unlock MQTT, make all reset, After select the device, turn Scan btn into Connect btn
1 : Click Connect btn   -> Lock MQTT
2 : when Connected   -> turn on the text blinking effect, turn Connect btn into Stop btn
"""
FLOW_CTRL_STATE = 0


# TK new UI
Log_text_val = ""
Btn_BLE_control_color = ["#5CADAD", "#02C874", "#FF2D2D"]
# 36 : TriAnswer packet size / CH
waveform_buffer_len = math.ceil((freq * 7) / 36) * 36
Signal_FRAME_LEN = waveform_buffer_len / freq
waveform_buffer = np.zeros((CH_NUM + 1, waveform_buffer_len))
waveform_buffer[waveform_buffer == 0] = np.nan
waveform_buffer_index = np.zeros(CH_NUM + 1, int)
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
    global client, FLAG_ALLOW_CONN, FLAG_SENSING_EN, MQTTClient, MQTT_ACT, MQTT_PWD, MQTT_IP, MQTT_PORT, MQTTMsg_Level, async_loop, FLOW_CTRL_STATE

    if FLOW_CTRL_STATE == 0:  # Before device selecting
        print("> Start scanning ...")
        log_update("> Start scanning ...")
        threading.Thread(target=_scan_thread, args=(async_loop,)).start()
    if FLOW_CTRL_STATE == 1:  # Try to connect to device
        log_update("> Start connecting ...")

        # Disable the setting views
        Combobox_BLE_Device["state"] = "disable"

        client = BleakClient(Target_device.address)
        FLAG_ALLOW_CONN = True
        FLAG_SENSING_EN = True
        print(f"> Start connecting {Target_device.address} ...")
        threading.Thread(target=_connect_to_device_thread, args=(async_loop,)).start()
        threading.Thread(target=turn_on_camera).start()
        threading.Thread(target=waveform_draw_task).start()  # draw waveform
    if FLOW_CTRL_STATE == 2:  # Try to disconnect to device
        FLAG_SENSING_EN = False
        FLOW_CTRL_STATE = 0


def device_selected(event):
    global Target_device, FLOW_CTRL_STATE
    FLOW_CTRL_STATE = 1
    Target_device = Device_list[Combobox_BLE_Device.current()]
    Btn_BLE_control["text"] = "Connect to Device"
    Btn_BLE_control["bg"] = Btn_BLE_control_color[FLOW_CTRL_STATE]


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
                f"{Device_list_advData[ind].local_name}({Device_list[ind].rssi})"
            )
        log_update("> Select the device you want to connect at the [drop-down] list")
        Combobox_BLE_Device["state"] = "readonly"
        Combobox_BLE_Device["value"] = dev_list_str
    else:
        log_update("> No TriAnswer Device Found !")


def detection_callback(device, advertisement_data):
    global Target_device
    if device is not None:
        if DEVICE_NAME_FILTER in f"{device}":
            allow_append = True
            for ind in range(len(Device_list)):
                if Device_list[ind].address == device.address:
                    allow_append = False
            if allow_append:
                Device_list.append(device)
                Device_list_advData.append(advertisement_data)


async def handle_rx_data(_: int, data: bytearray):
    global pkg_NO_pre, Device_SN_flag, SN_str, waveform_buffer_index, waveform_buffer, PPG_DC_GET_FLAG, PPG_DC

    # print('data len',len(data))
    # print('data',data)
    # BLE Data to CH 1 & 2 ECG Values
    record_time = time.time()
    ECG_temp_arr = np.zeros(PPG_VALID_LEN_INDEX)
    for ind in range(0, PPG_VALID_LEN_INDEX):
        ECG_temp_arr[ind] = (data[ind] & 0xFF) * 3600 / (2**8)  # Unit : mV

    # BLE Data to PPG Values
    PPG_DATA_INDEX_BASE = PPG_VALID_LEN_INDEX + 1
    PPG_VALID_LEN = round((data[PPG_VALID_LEN_INDEX] & 0xFF) / 4)
    # print('PPG_VALID_LEN',PPG_VALID_LEN)
    PPG_temp_arr = np.zeros((2, PPG_VALID_LEN))
    for ind in range(0, PPG_VALID_LEN):
        PPG_temp_arr[0][ind] = (data[PPG_DATA_INDEX_BASE + ind * 4] & 0xFF) + (
            data[PPG_DATA_INDEX_BASE + ind * 4 + 1] & 0xFF
        ) * 256
        PPG_temp_arr[1][ind] = (data[PPG_DATA_INDEX_BASE + ind * 4 + 2] & 0xFF) + (
            data[PPG_DATA_INDEX_BASE + ind * 4 + 3] & 0xFF
        ) * 256

    if not PPG_DC_GET_FLAG:
        PPG_DC = [PPG_temp_arr[0][0], PPG_temp_arr[1][0]]
        PPG_temp_arr = [PPG_temp_arr[0][1:], PPG_temp_arr[1][1:]]
        PPG_DC_GET_FLAG = True

    # print('ECG_temp_arr shape=',ECG_temp_arr.shape)
    # print('PPG_temp_arr shape=',PPG_temp_arr.shape)

    # Waveform Processing (the right the newer)
    wave_temp_array = [[], [], [], []]
    length = []
    for ch_ind in range(0, 4):
        # Prepare the drawing data per channel (先全部填入，繪圖前才做降頻)

        if ch_ind < 2:  # CH1~2
            wave_temp_array[ch_ind] = ECG_temp_arr[ch_ind::2]
        else:  # CH3~4 : 10倍頻至1000Hz再降頻
            wave_temp_array[ch_ind] = (
                np.repeat(PPG_temp_arr[ch_ind - 2], CH_SpeedMode[ch_ind])
                - PPG_DC[ch_ind - 2]
            )

        # print('wave_temp_array ' ,f'{ch_ind} shape=',wave_temp_array[ch_ind].shape)
        # Write to file
        if (
            Btn_Record_State == 1
        ):  # when record bottom trigger and the function change the state it will start write  txt file
            # start_recording_2 = time.time() ## test thread
            # print("start_save_file",start_recording_2)

            # print('ecg record time=',enter_time)
            if ch_ind == 0:  # ecg
                length.append(len(wave_temp_array[ch_ind][:: CH_SpeedMode[ch_ind]]))
                for ele in wave_temp_array[ch_ind][:: CH_SpeedMode[ch_ind]]:
                    TA_FileHandle[ch_ind].write(f"{ele}\n")
            if ch_ind == 2:  # ppgir
                counter = 0
                length.append(len(wave_temp_array[ch_ind][:: CH_SpeedMode[ch_ind]]))
                for ele in wave_temp_array[ch_ind][:: CH_SpeedMode[ch_ind]]:
                    TA_FileHandle[ch_ind].write(f"{ele}\n")
                    counter += 1
                    """
                    # deal ppg high peak
                    if(abs(ele)>7000):
                        print('time',record_time)
                        print('ir spick=',ele)
                        TA_FileHandle[ch_ind].write(f'{ele}\n')
                        counter+=1
                        #TA_FileHandle[ch_ind].write(f'{ele}\n')
                        continue
                    else:
                        TA_FileHandle[ch_ind].write(f'{ele}\n')
                        counter+=1
                    """
                # length.append(counter)
            if ch_ind == 3:  # ppgr
                counter = 0
                length.append(len(wave_temp_array[ch_ind][:: CH_SpeedMode[ch_ind]]))
                for ele in wave_temp_array[ch_ind][:: CH_SpeedMode[ch_ind]]:
                    TA_FileHandle[ch_ind].write(f"{ele}\n")
                    counter += 1
                    # deal ppg high peak
                    """
                    if(abs(ele)>7000):
                        print('r spick=',ele)
                        TA_FileHandle[ch_ind].write(f'{ele}\n')
                        counter+=1
                        continue
                    else:
                        TA_FileHandle[ch_ind].write(f'{ele}\n')
                        counter+=1
                    """
                # length.append(counter)

        if (waveform_buffer_index[ch_ind] + len(wave_temp_array[ch_ind])) <= len(
            waveform_buffer[ch_ind]
        ):  # buffer not full
            waveform_buffer[ch_ind][
                waveform_buffer_index[ch_ind] : (
                    waveform_buffer_index[ch_ind] + len(wave_temp_array[ch_ind])
                )
            ] = wave_temp_array[ch_ind]
            waveform_buffer_index[ch_ind] = waveform_buffer_index[ch_ind] + len(
                wave_temp_array[ch_ind]
            )
        else:
            waveform_buffer[ch_ind][: -len(wave_temp_array[ch_ind])] = waveform_buffer[
                ch_ind
            ][len(wave_temp_array[ch_ind]) :]
            waveform_buffer[ch_ind][-len(wave_temp_array[ch_ind]) :] = wave_temp_array[
                ch_ind
            ]
    # record time step
    if Btn_Record_State == 1:
        TA_FileHandle[1].write(f"{record_time},{length[0]},{length[1]},{length[2]}\n")


def _connect_to_device_thread(async_loop):
    async_loop.run_until_complete(connect_to_device())


async def connect_to_device():
    global conn_connected, client, Target_device, MQTTClient, FLOW_CTRL_STATE, waveform_buffer, waveform_buffer_index, PPG_DC_GET_FLAG

    while FLAG_SENSING_EN:

        if FLAG_ALLOW_CONN:
            try:
                await client.connect()
                await client.get_services()
                conn_connected = client.is_connected()
                print("> Device connected !")

                if conn_connected:
                    FLOW_CTRL_STATE = 2
                    Btn_Record["state"] = "normal"
                    Btn_Record["bg"] = "#ef5350"
                    Btn_BLE_control["text"] = "Stop Connection"
                    Btn_BLE_control["bg"] = Btn_BLE_control_color[FLOW_CTRL_STATE]
                    log_update(
                        "> Connect to TriAnswer and try to initiate the measurement ..."
                    )
                    # waveform_draw_task()

                    await client.start_notify(DATA_VAL_UUID, handle_rx_data)
                    log_update("> Sensing ...")

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
    log_update(f"> [BLE] Try to disconnect...")
    await client.disconnect()
    # closeRecordFile()  # i think if disconnect and forget stop record will use it to close file

    if not FLAG_WINDOW_CLOSE:
        conn_connected = False
        client = None
        log_update(f"> [BLE] {Target_device.address} disconnected...")
        Target_device = None
        MQTTClient = None
        window.after(300, None)
        # log_update(f'> Procedure is Stopped ...')
        # window.after(100, None)
        FLOW_CTRL_STATE = 0
        log_update(f"> Click on the [Scan] button to connect to TriAnswer Device...")
        Btn_BLE_control["text"] = "Scan"
        Btn_BLE_control["bg"] = Btn_BLE_control_color[FLOW_CTRL_STATE]
        Combobox_BLE_Device["value"] = []
        Combobox_BLE_Device.set("")
        Combobox_BLE_Device["state"] = "disable"
        Btn_Record["state"] = "disable"
        Btn_Record["bg"] = "grey"
        PPG_DC_GET_FLAG = False

        # Drawing param reset
        reset_all_figure()

    else:
        window.quit()


def reset_all_figure():
    global waveform_buffer, waveform_buffer_index
    waveform_buffer = np.zeros((CH_NUM + 1, waveform_buffer_len))
    waveform_buffer[waveform_buffer == 0] = np.nan
    waveform_buffer_index = np.zeros(CH_NUM + 1, int)

    for ch_num in range(0, 2):
        mat_ax[ch_num].clear()
        number = ch_num
        if ch_num == 1:
            number = 2
        mat_ax[ch_num].plot(
            time_array[:: CH_SpeedMode[number]],
            waveform_buffer[number][:: CH_SpeedMode[number]],
            color="r",
            linewidth=1,
            alpha=0.7,
        )

        # axis setting
        mat_ax[ch_num].set_xlim(0, Signal_FRAME_LEN)
        mat_ax[ch_num].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
        mat_ax[ch_num].set_xticklabels([])
        mat_ax[ch_num].grid(axis="x")
        mat_ax[ch_num].yaxis.set_visible(False)
        mat_ax[ch_num].set_yticks([])

    mat_ax[0].set_title("ECG")
    mat_ax[1].set_title("PPG")
    canvas.draw()


def reset_figure(ch_num):
    global waveform_buffer, waveform_buffer_index
    waveform_buffer[ch_num] = np.zeros(waveform_buffer_len)
    waveform_buffer[ch_num][waveform_buffer[ch_num] == 0] = np.nan
    waveform_buffer_index[ch_num] = 0

    mat_ax[ch_num].clear()
    mat_ax[ch_num].plot(
        time_array[:: CH_SpeedMode[ch_num]],
        waveform_buffer[ch_num][:: CH_SpeedMode[ch_num]],
        color="r",
        linewidth=1,
        alpha=0.7,
    )

    # axis setting
    mat_ax[ch_num].set_xlim(0, Signal_FRAME_LEN)
    mat_ax[ch_num].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
    mat_ax[ch_num].set_xticklabels([])
    mat_ax[ch_num].grid(axis="x")
    mat_ax[ch_num].yaxis.set_visible(False)
    mat_ax[ch_num].set_yticks([])
    canvas.draw()


def log_blinking_show():
    Log_text["text"] = Log_text_val


def log_update(input_str):
    global Log_text_val
    Log_text_val = input_str
    Log_text["text"] = ""
    window.after(50, log_blinking_show)


def waveform_draw_task():
    global conn_connected, client, Target_device, MQTTClient, FLOW_CTRL_STATE, waveform_buffer, waveform_buffer_index, PPG_DC_GET_FLAG, FLAG_SENSING_EN

    while FLAG_SENSING_EN:
        for ch_num in range(0, 2):
            mat_ax[ch_num].clear()

            if ch_num < 1:  # delete ch2
                fs = 1000
                taps = firwin(70, [0.5 / (fs / 2), 55 / (fs / 2)], pass_zero=False)
                plot_signal = filtfilt(
                    taps,
                    1.0,
                    np.double(waveform_buffer[ch_num][:: CH_SpeedMode[ch_num]]),
                )
                # print('ECG plot shape : ',waveform_buffer[ch_num][::CH_SpeedMode[ch_num]].shape)
                mat_ax[ch_num].plot(
                    time_array[:: CH_SpeedMode[ch_num]],  # CH0
                    plot_signal,
                    color="r",
                    linewidth=1,
                    alpha=0.7,
                )
            else:
                # CH3
                # 紅外光
                """
                mat_ax[ch_num].plot(time_array[::CH_SpeedMode[ch_num+1]],#1+1=CH2
                                    waveform_buffer[ch_num +
                                                    1][::CH_SpeedMode[ch_num+1]],
                                    color='r', linewidth=1, alpha=0.7)
                """
                # CH4
                # 紅光
                fs = 100
                taps = firwin(7, [0.5 / (fs / 2), 10 / (fs / 2)], pass_zero=False)
                # waveform_buffer[ch_num +2][::CH_SpeedMode[ch_num + 2]]=waveform_buffer[ch_num +2][::CH_SpeedMode[ch_num + 2]]*-1
                plot_signal_2 = (
                    filtfilt(
                        taps,
                        1.0,
                        np.double(
                            waveform_buffer[ch_num + 2][:: CH_SpeedMode[ch_num + 2]]
                        ),
                    )
                    * -1
                )
                # print('ppg plot shape : ',waveform_buffer[ch_num +2][::CH_SpeedMode[ch_num + 2]].shape)
                mat_ax[ch_num].plot(
                    time_array[:: CH_SpeedMode[ch_num + 2]],  # CH3
                    plot_signal_2,
                    color="b",
                    linewidth=1,
                    alpha=0.7,
                )
            # axis setting
            mat_ax[ch_num].set_xlim(0, Signal_FRAME_LEN)
            mat_ax[ch_num].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
            mat_ax[ch_num].set_xticklabels([])
            mat_ax[ch_num].grid(axis="x")
            mat_ax[ch_num].yaxis.set_visible(False)
            mat_ax[ch_num].set_yticks([])
        mat_ax[0].set_title("ECG")
        mat_ax[1].set_title("PPG")
        # update figure
        canvas.draw()
        time.sleep(1)

    # if FLOW_CTRL_STATE == 2:  # when connected
    #    window.after(waveform_draw_period, waveform_draw_task)


class ipcamCapture:
    def __init__(self):
        global camera_num, gain, exposure
        self.Frame = []
        self.status = False
        self.isstop = False
        self.connect = False
        # 攝影機連接 # 1 外接
        self.capture = cv2.VideoCapture(camera_num)
        if not self.capture.isOpened():
            print("just have one camera")
            self.capture = cv2.VideoCapture(0)

        # 先讀一張，改值才有效
        self.status, self.Frame = self.capture.read()
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.exp = exposure
        self.gain = gain
        exp_success = self.capture.set(cv2.CAP_PROP_EXPOSURE, self.exp)
        gain_success = self.capture.set(cv2.CAP_PROP_GAIN, self.gain)
        exposure_entry.insert(0, str(""))
        gain_entry.insert(0, str(""))
        exposure_entry.insert(0, str(self.exp))
        gain_entry.insert(0, str(self.gain))
        # 關掉相機自動的功能
        # exp = self.capture.get(cv2.CAP_PROP_EXPOSURE)
        # wb = self.capture.get(cv2.CAP_PROP_WB_TEMPERATURE)
        # fc = self.capture.get(cv2.CAP_PROP_FOCUS)
        # self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)  # 關掉自動白平衡 0
        # self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # self.capture.set(cv2.CAP_PROP_EXPOSURE, -4)
        # self.capture.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        # self.capture.set(cv2.CAP_PROP_FOCUS, fc)
        # self.capture.set(cv2.CAP_PROP_BRIGHTNESS,100)  #控制亮度
        if not exp_success:
            p_exp = False
        else:
            p_exp = exposure
        if not gain_success:
            p_gain = False
        else:
            p_gain = gain
        exposure_value_label.config(text="曝光值: {}".format(p_exp))
        gain_value_label.config(text="增益值: {}".format(p_gain))
        if not self.capture.isOpened():
            print("Cannot open camera")
            exit()
        else:
            print("turn on camera")
            self.connect = True

    # use to set exp
    def set_parameter(self):
        # global exposure_entry,gain_entry,exposure_value_label,gain_value_label

        self.exp = int(exposure_entry.get())
        self.gain = int(gain_entry.get())
        exp_success = self.capture.set(cv2.CAP_PROP_EXPOSURE, self.exp)
        gain_success = self.capture.set(cv2.CAP_PROP_GAIN, self.gain)
        if not exp_success:
            self.exp = False
        if not gain_success:
            self.gain = False
        exposure_value_label.config(text="曝光值: {}".format(self.exp))
        gain_value_label.config(text="增益值: {}".format(self.gain))

    def in_set_parameter(self, exp, gain):
        # global exposure_entry,gain_entry,exposure_value_label,gain_value_label
        self.exp = exp
        self.gain = gain
        exp_success = self.capture.set(cv2.CAP_PROP_EXPOSURE, self.exp)
        gain_success = self.capture.set(cv2.CAP_PROP_GAIN, self.gain)
        if not exp_success:
            self.exp = False
        if not gain_success:
            self.gain = False
        exposure_value_label.config(text="曝光值: {}".format(self.exp))
        gain_value_label.config(text="增益值: {}".format(self.gain))

    def start(self):
        # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print("ipcam started!")
        threading.Thread(
            target=self.queryframe, daemon=True, args=()
        ).start()  # daemon=True python 檔結束執行，強制結束

    def stop(self):
        # 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print("ipcam stopped!")
        self.capture.release()

    def getframe(self):
        # 當有需要影像時，再回傳最新的影像。
        return self.Frame.copy()

    def queryframe(self):
        global video, imgpath, save_time_complete
        while not self.isstop:
            self.status, self.Frame = self.capture.read()
            if not self.status:
                print("Can't receive frame (stream end?). Exiting ...")
            else:
                # 記錄影片
                counter = 0
                start_recording = time.time()
                record = 0
                temp_time_step = list()
                while Btn_Record_State:
                    record = 1
                    temp_time_step.append(time.time())
                    self.status, self.Frame = self.capture.read()
                    if self.status == True:
                        # video record
                        # video.write(self.Frame)
                        # img record
                        counter += 1
                        if counter < 10:
                            name = f"0000{counter}"
                        elif counter < 100:
                            name = f"000{counter}"
                        elif counter < 1000:
                            name = f"00{counter}"
                        elif counter < 10000:
                            name = f"0{counter}"
                        elif counter < 100000:  # 10000-99999
                            name = f"{counter}"
                        else:
                            print("recording stop,too long")
                            break
                        cv2.imwrite(
                            f"{imgpath}/{name}.png",
                            self.Frame,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0],
                        )
                out_time = time.time()
                if record == 1:
                    file = open(f"{imgpath}/time_step.txt", "w")
                    file.write(f"actually record time = {out_time-start_recording}\n")
                    file.write(str(temp_time_step))
                    file.close()
                    print("counter=", counter)
                    print("start record video time=", start_recording)
                    print("stop record video time=", out_time)
                    print("actual record time", out_time - start_recording)
                    save_time_complete = True
                # record=0

    def automatic_adjust_light(self):
        complete = False
        counter = 0
        while not complete:
            counter += 1
            adjust_frame = []
            while len(adjust_frame) < 60:
                frame = self.getframe()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                adjust_frame.append(frame)
                time.sleep(0.01)
            frames = np.asarray(adjust_frame)
            new, save_img, no_face = crop_face_resize(
                frames=frames,
                use_dynamic_detection=True,
                detection_freq=2,
                use_median_box=True,
                width=72,
                height=72,
            )
            if no_face:
                adjust_button.config(text=f"{counter} no_face")
                adjust_button.update()
            else:
                standardized, mean, std, L = standardized_data(new)
                if 155 <= mean and mean <= 185:
                    complete = True
                    exposure_value_label.config(text="曝光值: {}".format(self.exp))
                    gain_value_label.config(text="增益值: {}".format(self.gain))
                    mean_value_label.config(text="mean: {}".format(mean))
                    adjust_button.config(text=f"adjust")
                    adjust_button.update()
                    gain_value_label.update()
                    mean_value_label.update()
                    exposure_value_label.update()
                elif 155 > mean:  # 加光
                    diff = int(175 - mean)
                    exp = diff // 60
                    if exp > 0:
                        self.in_set_parameter(self.exp + exp, self.gain)
                    else:
                        self.in_set_parameter(self.exp, self.gain + diff)
                    adjust_button.config(
                        text=f"{counter},adjust exp={self.exp},gain={self.gain},ori mean={mean}"
                    )
                    adjust_button.update()
                    print(
                        f"{counter},adjust exp={self.exp},gain={self.gain},ori mean={mean}"
                    )
                else:  # 減光
                    diff = int(mean - 175)
                    exp = diff // 60
                    if exp > 0:
                        self.in_set_parameter(self.exp - exp, self.gain)
                    else:
                        new_gain = self.gain - diff
                        if new_gain < 0:
                            self.in_set_parameter(self.exp - 1, 64)
                        else:
                            self.in_set_parameter(self.exp, new_gain)
                    adjust_button.config(
                        text=f"{counter},adjust exp={self.exp},gain={self.gain},ori mean={mean}"
                    )
                    adjust_button.update()
                    print(
                        f"{counter},adjust exp={self.exp},gain={self.gain},ori mean={mean}"
                    )
            time.sleep(1)
            if counter > 10:
                complete = True


# use to set camera parameter
def apply_settings():
    global ipcam
    if ipcam != None:
        ipcam.set_parameter()


def adjust_light():
    global ipcam
    if ipcam != None:
        ipcam.automatic_adjust_light()


def turn_on_camera():
    print("enter function turn_on_camera")
    global camera, video, FLAG_SENSING_EN, Btn_Record_State, ipcam
    ipcam = ipcamCapture()
    ipcam.start()
    # 讓相機初始設定
    while not ipcam.connect:
        time.sleep(1)

    # 只要是sensing 時，照相機也會一直開著
    while FLAG_SENSING_EN:
        # 擷取影像
        frame = ipcam.getframe()
        if ipcam.status:
            try:
                frame = cv2.resize(
                    frame, (image_width, image_height), interpolation=cv2.INTER_AREA
                )  # 有可能第一張收不到frame
            except:
                print("can not get frame")
                continue
            # resize image to 640*480
            frame = cv2.flip(frame, 1)  # 左右翻轉
            # add rectangle
            cv2.rectangle(
                frame,
                (
                    int(5 * frame.shape[1] / 10 + frame.shape[1] / 20),
                    int(frame.shape[0] / 5),
                ),
                (int(9 * frame.shape[1] / 10), int(4 * frame.shape[0] / 5)),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Hand",
                (
                    int(5 * frame.shape[1] / 10 + frame.shape[1] / 20),
                    int(frame.shape[0] / 5) - 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                frame,
                (int(1 * frame.shape[1] / 10), int(1 * frame.shape[0] / 6)),
                (int(5 * frame.shape[1] / 10), int(5 * frame.shape[0] / 6)),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Face",
                (int(1 * frame.shape[1] / 10), int(1 * frame.shape[0] / 6) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            # show frame on tk
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(frame)
            # ilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=pilImage)
            movieLabel.config(image=imgtk)
            movieLabel.image = imgtk
            # window.update()  # update 照片

            start_recording = time.time()
            record = 0
            while Btn_Record_State:
                record = 1
                now_time = time.time()
                frame = ipcam.getframe()
                frame = cv2.resize(
                    frame, (image_width, image_height), interpolation=cv2.INTER_AREA
                )
                frame = cv2.flip(frame, 1)  # 左右翻轉
                # 放時間
                pos = (450, 10)
                x, y = pos
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 2
                font_thickness = 1
                text = f"time{str(int(now_time-start_recording))}s"
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_w, text_h = text_size
                cv2.rectangle(frame, pos, (x + text_w, y + text_h), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    text,
                    (x, y + text_h + font_scale - 1),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                )
                # 放框框
                cv2.rectangle(
                    frame,
                    (
                        int(5 * frame.shape[1] / 10 + frame.shape[1] / 20),
                        int(frame.shape[0] / 5),
                    ),
                    (int(9 * frame.shape[1] / 10), int(4 * frame.shape[0] / 5)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Hand",
                    (
                        int(5 * frame.shape[1] / 10 + frame.shape[1] / 20),
                        int(frame.shape[0] / 5) - 5,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    frame,
                    (int(1 * frame.shape[1] / 10), int(1 * frame.shape[0] / 6)),
                    (int(5 * frame.shape[1] / 10), int(5 * frame.shape[0] / 6)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Face",
                    (int(1 * frame.shape[1] / 10), int(1 * frame.shape[0] / 6) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                # show frame on tk
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pilImage = Image.fromarray(frame)
                # pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
                imgtk = ImageTk.PhotoImage(image=pilImage)
                movieLabel.config(image=imgtk)
                movieLabel.image = imgtk
            out_time = time.time()
            if record == 1:
                print("actual record time of show", out_time - start_recording)
            record = 0
    # 釋放該攝影機裝置
    print("turn off camera")
    ipcam.stop()
    print("leave function turn_on_camera")
    # 放logo
    frame = cv2.imread("./logo.jpg")
    frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(frame)
    # ilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=pilImage)
    movieLabel.config(image=imgtk)
    movieLabel.image = imgtk


def createRecordFile():
    global TA_FileHandle, video, imgpath  # add global variable video

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
    time_str = now_time.strftime("%Y_%m_%d_%H_%M_%S")
    file_dir = f"{TriAnswer_Record_Dir}/" + time_str
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    for ch_num in range(0, 3):
        if ch_num == 0:  # CH1
            FileName = f"ECG.csv"
            TA_FileHandle[ch_num] = open(os.path.join(file_dir, FileName), "w")
        elif ch_num == 1:  # CH2
            FileName = f"ECG_PPG_time_step.txt"
            # 不存ch2 改存時間
            TA_FileHandle[ch_num] = open(os.path.join(file_dir, FileName), "w")
        else:
            # CH3
            FileName = f"PPG_IR.csv"
            TA_FileHandle[ch_num] = open(os.path.join(file_dir, FileName), "w")
            # CH4
            FileName = f"PPG_R.csv"
            TA_FileHandle[ch_num + 1] = open(os.path.join(file_dir, FileName), "w")

    # add video file
    # video_name = file_dir + r'/video.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'I420') #I420 XVID
    # video = cv2.VideoWriter(video_name, fourcc, 30, (640, 480))
    # add img file
    imgpath = os.path.join(file_dir, "img")
    dirIsExist = os.path.exists(imgpath)
    if not dirIsExist:
        os.makedirs(imgpath)
    return file_dir


def closeRecordFile(file_dir: str):
    global TA_FileHandle, Btn_Record_State

    Btn_Record_State = 0
    if (
        TA_FileHandle[0] is not None
        or TA_FileHandle[1] is not None
        or TA_FileHandle[2] is not None
        or TA_FileHandle[3] is not None
    ):
        for ch_num in range(0, 4):
            if ch_num != 1:
                TA_FileHandle[ch_num].close()

        TA_FileHandle = [None, None, None, None]

    # Enable the setting frame
    Btn_Record["text"] = "Start\nREC"
    Btn_Record["bg"] = "#ef5350"

    threading.Thread(target=normalize, daemon=False, args=(file_dir,)).start()
    """
    chk_CH1.configure(state='normal')
    chk_CH2.configure(state='normal')
    chk_CH3.configure(state='normal')
    for ch_ind in range(0, 3):
        if CH_EN[ch_ind].get():
            for childView in frm_Ch[ch_ind].winfo_children():
                childView.configure(state='normal')
    """


def normalize(file_dir):
    global save_time_complete
    while not save_time_complete:
        print("wait save time and to normalize")

    ecg_path = f"{file_dir}/ECG.csv"
    img_path = f"{file_dir}/img/*.png"
    img = glob.glob(img_path)
    signal = pd.read_csv(ecg_path)
    signal = signal.values.reshape(1, -1).squeeze()
    print(signal.shape)
    signal = signal.tolist()
    print(len(signal))
    print(f"calculate record video time  =  {len(img)//30}")
    print(f"calculate record video time  =  {len(signal)//1000}")
    print("Please Check the duration ")


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
    global Btn_Record_State, save_file_dir, save_time_complete

    if Btn_Record_State == 0:
        save_file_dir = createRecordFile()  # just creat file
        Btn_Record["text"] = "STOP\nREC"
        Btn_Record["bg"] = "#96d1b1"
        start_recording = time.time()
        Btn_Record_State = 1  # set to one start write data fo file
        save_time_complete = False
        print("set_record_state=1 time=", start_recording)
    elif Btn_Record_State == 1:
        closeRecordFile(save_file_dir)


## adjust lighting function


def detect(frame):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_zone = detector.detectMultiScale(frame)
    No_face = False
    if len(face_zone) < 1:
        No_face = True
        # print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        # Find the index of the largest face zone
        # The face zones are boxes, so the width and height are the same
        max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
        face_box_coor = face_zone[max_width_index]
        # print("Warning: More than one faces are detected. Only cropping the biggest one.")
    else:
        face_box_coor = face_zone[0]
    use_larger_box = True
    larger_box_coef = 0.6
    y_scale = 0.4
    if use_larger_box:
        face_box_coor[0] = max(
            0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2]
        )
        face_box_coor[1] = max(
            0, face_box_coor[1] - (larger_box_coef - 1.0) * face_box_coor[3]
        )
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = y_scale * face_box_coor[3]
    return face_box_coor, No_face


def crop_face_resize(
    frames, use_dynamic_detection, detection_freq, use_median_box, width, height
):
    # Face Cropping
    if use_dynamic_detection:
        num_dynamic_det = ceil(frames.shape[0] / detection_freq)
    else:
        num_dynamic_det = 1
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    counter = 0
    for idx in range(num_dynamic_det):
        bbx, no_face = detect(frames[detection_freq * idx])
        face_region_all.append(bbx)
        if no_face:
            counter += 1
        # print('frame ',detection_freq * idx, ' bbx: ',face_region_all[idx])
    cv2.destroyAllWindows()
    face_region_all = np.asarray(face_region_all, dtype="int")
    if use_median_box:
        # Generate a median bounding box based on all detected face regions
        face_region_median = np.median(face_region_all, axis=0).astype("int")
    print("first:", face_region_all[0])
    print("median:", face_region_median)

    face_region = face_region_median
    save_img = frames[0][
        max(face_region[1], 0) : min(
            face_region[1] + face_region[3], frames[0].shape[0]
        ),
        max(face_region[0], 0) : min(
            face_region[0] + face_region[2], frames[0].shape[1]
        ),
    ]
    # Frame Resizing
    resized_frames = np.zeros((frames.shape[0], height, width, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
            reference_index = i // detection_freq
        else:  # use the first region obtrained from the first frame.
            reference_index = 0
        if use_median_box:
            face_region = face_region_median
        else:
            face_region = face_region_all[reference_index]
        frame = frame[
            max(face_region[1], 0) : min(
                face_region[1] + face_region[3], frame.shape[0]
            ),
            max(face_region[0], 0) : min(
                face_region[0] + face_region[2], frame.shape[1]
            ),
        ]
        resized_frames[i] = cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_AREA
        )

    return resized_frames, save_img, counter > 8  # (True no face detect)


def standardized_data(data):
    """Z-score standardization for video data."""
    mean = np.mean(data)
    std = np.std(data)
    print(f"standardized mean：{mean} std：{std}")
    R_mean = np.mean(data[:, :, :, 0])
    G_mean = np.mean(data[:, :, :, 1])
    B_mean = np.mean(data[:, :, :, 2])
    L = 0.213 * R_mean + 0.715 * G_mean + 0.072 * B_mean
    print(f"Brightness = {L}")
    data = data - mean
    data = data / std

    data[np.isnan(data)] = 0
    return data, mean, std, L


if __name__ == "__main__":

    # Initiate TK GUI
    window = tk.Tk()
    window.title("Yutech - TriAnswer Controller")
    window.geometry("1600x850")  # 注意是string，而不是數字 #inital window size
    window.resizable(width=False, height=False)
    font = ("Arial", 15, "bold")  # 字體大小
    figsize = (9, 6)  # 繪圖大小

    frm_BLE = tk.LabelFrame(
        master=window, text="TriAnswer Setting & Control", padx=5, pady=5
    )
    CH_SpeedMode = [int(1), int(1), int(10), int(10)]  #::等間距取值
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
    frm_BleCon = tk.LabelFrame(master=frm_BLE, text="BLE Connection", padx=5, pady=5)

    # BLE Device
    Combobox_BLE_Device = ttk.Combobox(master=frm_BleCon, width=28, state="disable")
    Combobox_BLE_Device.bind("<<ComboboxSelected>>", device_selected)
    Combobox_BLE_Device.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    # BLE Control BTN
    Btn_BLE_control = tk.Button(
        master=frm_BleCon,
        text="Scan",
        width=10,
        font=font,
        bg=Btn_BLE_control_color[FLOW_CTRL_STATE],
        fg="yellow",
        command=FLOW_CTRL_BTN_action,
    )
    Btn_BLE_control.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    # Record BTN
    Btn_Record = tk.Button(
        master=frm_BleCon,
        text="Start\nREC",
        width=6,
        state="disable",
        font=font,
        bg="grey",
        fg="white",
        command=Btn_Record_Ctrl,
    )
    Btn_Record.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")
    Btn_Record_State = 0

    frm_BleCon.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    frm_BLE.grid(row=0, column=1, padx=5, sticky="nsew")

    # add movie
    # canvas_v = tk.Canvas(master=frm_BLE, width=image_width, height=image_height)  # 加入 Canvas 畫布
    # canvas_v.place(x=0,y=0)
    # canvas.grid()
    movieLabel = tk.Label(master=frm_BLE)
    movieLabel.grid()
    # 先放一張logo
    frame = cv2.imread("./logo.jpg")
    frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(frame)
    # ilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=pilImage)
    movieLabel.config(image=imgtk)
    movieLabel.image = imgtk
    # Figure block
    number_of_figure = 2
    frm_fig = tk.LabelFrame(master=window, text="Waveform Monitor", padx=5, pady=5)
    time_array = np.arange(waveform_buffer_len) / freq
    mat_fig, mat_ax = plt.subplots(
        number_of_figure, 1, figsize=figsize, dpi=100, constrained_layout=True
    )

    mat_fig.patch.set_facecolor("#F0F0F0")
    mat_fig.patch.set_alpha(1.0)

    for i in range(0, number_of_figure):
        mat_ax[i].set_facecolor("white")
        mat_ax[i].set_xlim(0, Signal_FRAME_LEN)
        mat_ax[i].set_xticks(np.arange(0, Signal_FRAME_LEN, step=0.2))
        mat_ax[i].set_xticklabels([])
        mat_ax[i].grid(axis="x")
        mat_ax[i].yaxis.set_visible(False)
        mat_ax[i].set_yticks([])

    mat_ax[0].set_title("ECG")
    mat_ax[1].set_title("PPG")
    canvas = FigureCanvasTkAgg(mat_fig, master=frm_fig)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    frm_fig.grid(row=0, column=0, padx=5, sticky="nsew")

    # Log block
    frm_log = tk.LabelFrame(master=window, text="Log", padx=5, pady=5)
    Log_text = tk.Label(master=frm_log, bg="#D1E9E9", anchor="w")
    Log_text["text"] = f"> Click on the [Scan] button to connect to TriAnswer Device..."
    Log_text.pack(fill="x")
    frm_log.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

    # 创建曝光调整框架
    exposure_frame = ttk.LabelFrame(master=window, text="camera 調整")
    exposure_frame.grid(
        row=2, column=1, columnspan=1, padx=10, pady=5, sticky=tk.W + tk.E
    )

    exposure_entry = ttk.Entry(exposure_frame)
    exposure_entry.grid(row=0, column=0, padx=10, pady=5)

    exposure_value_label = ttk.Label(exposure_frame, text="曝光值: ")
    exposure_value_label.grid(row=1, column=0, padx=10, pady=5)

    # 创建增益调整框架
    gain_entry = ttk.Entry(exposure_frame)
    gain_entry.grid(row=0, column=1, padx=10, pady=5)

    gain_value_label = ttk.Label(exposure_frame, text="增益值: ")
    gain_value_label.grid(row=1, column=1, padx=10, pady=5)
    mean_value_label = ttk.Label(exposure_frame, text="mean: ")
    mean_value_label.grid(row=1, column=2, padx=10, pady=5)
    apply_button = ttk.Button(
        exposure_frame, text="應用", command=apply_settings
    )  # command=self.apply_settings
    apply_button.grid(
        row=2, column=0, columnspan=1, padx=10, pady=5, sticky=tk.W + tk.E
    )
    adjust_button = ttk.Button(
        exposure_frame, text="adust_light", command=adjust_light
    )  # command=self.apply_settings
    adjust_button.grid(
        row=2, column=1, columnspan=1, padx=10, pady=5, sticky=tk.W + tk.E
    )

    # Asyncio
    async_loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    window.protocol("WM_DELETE_WINDOW", on_WinClosing)
    window.mainloop()
