import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class CameraControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("相機控制程式")

        # 创建曝光调整框架
        exposure_frame = ttk.LabelFrame(self.root, text="曝光調整")
        exposure_frame.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W+tk.E)

        self.exposure_entry = ttk.Entry(exposure_frame)
        self.exposure_entry.grid(row=0, column=0, padx=10, pady=5)

        self.exposure_value_label = ttk.Label(exposure_frame, text="曝光值: ")
        self.exposure_value_label.grid(row=1, column=0, padx=10, pady=5)

        # 创建增益调整框架
        gain_frame = ttk.LabelFrame(self.root, text="增益調整")
        gain_frame.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W+tk.E)

        self.gain_entry = ttk.Entry(gain_frame)
        self.gain_entry.grid(row=0, column=0, padx=10, pady=5)

        self.gain_value_label = ttk.Label(gain_frame, text="增益值: ")
        self.gain_value_label.grid(row=1, column=0, padx=10, pady=5)
        
        # 创建preprocessing image information
        pre_img_info = ttk.LabelFrame(self.root, text="preprocessing img information")
        pre_img_info.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W+tk.E)
        self.mean = ttk.Label(pre_img_info, text="standard mean： ")
        self.mean.grid(row=0, column=0, padx=10, pady=5)
        self.std = ttk.Label(pre_img_info, text="standard std： ")
        self.std.grid(row=1, column=0, padx=10, pady=5)
        self.b = ttk.Label(pre_img_info, text="standard brightness： ")
        self.b.grid(row=2, column=0, padx=10, pady=5)
        

        # 创建显示相机画面的标签
        self.camera_label = ttk.Label(self.root)
        self.camera_label.grid(row=0, column=1, rowspan=3, padx=10, pady=5)
        # 创建播放相机画面的标签
        #self.playback_label = ttk.Label(self.root)
        #self.playback_label.grid(row=0, column=2, rowspan=2, padx=10, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.tight_layout()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=3, padx=10, pady=5)  # 使用grid()方法放置在网格中

        # 创建按钮
        self.apply_button = ttk.Button(self.root, text="應用", command=self.apply_settings)
        self.apply_button.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)

        self.record_button = ttk.Button(self.root, text="Record", command=self.record_toggle)
        self.record_button.grid(row=4, column=0,columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 获取相机的曝光和增益值
        self.original_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        self.original_gain = self.cap.get(cv2.CAP_PROP_GAIN)
        
        print(self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0))
        # 将原始值显示在界面上
        self.exposure_entry.insert(0, str(int(self.original_exposure)))
        self.gain_entry.insert(0, str(int(self.original_gain)))

        self.is_recording = False
        self.recorded_frames = []
        self.record_start_time = None  # 初始化錄製開始時間
        
        self.cTime=0
        self.pTime=0
        self.show_camera_feed()
        

        # 绑定析构函数
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_camera_feed(self):
        ret, frame = self.cap.read()
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 300))
            cv2.putText(frame, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            if self.is_recording and self.record_start_time is not None:
                elapsed_time = datetime.now() - self.record_start_time
                timestamp = str(elapsed_time)[:-7]  # 截取時間長度
                frame_with_timestamp = self.add_timestamp(frame.copy(), timestamp)
            else:
                frame_with_timestamp = frame

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_with_timestamp),master=self.root)
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo

            if self.is_recording:
                self.recorded_frames.append(frame)

            self.camera_label.after(10, self.show_camera_feed)

    def apply_settings(self):
        print(type(self.exposure_entry.get()))
        exposure_value = int(self.exposure_entry.get())
        gain_value = int(self.gain_entry.get())
        # 设置相机的曝光和增益值
        print(exposure_value)
        #print('CAP_PROP_AUTO_EXPOSURE',self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        #print('CAP_PROP_AUTO_WB',self.cap.get(cv2.CAP_PROP_AUTO_WB))
        #print('CAP_PROP_AUTOFOCUS',self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
        #print('close wb',self.cap.set(cv2.CAP_PROP_AUTO_WB,0)) # 關掉自動白平衡 0
        #print('close expo',self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0))
        #print('close focus',self.cap.set(cv2.CAP_PROP_AUTOFOCUS , 0))
        print(self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value))
        print(self.cap.set(cv2.CAP_PROP_GAIN, gain_value))
        # 更新标签显示的值
        self.exposure_value_label.config(text="曝光值: {}".format(exposure_value))
        self.gain_value_label.config(text="增益值: {}".format(gain_value))

    def record_toggle(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_start_time = datetime.now()  # 開始錄製時的時間
            self.record_button.config(text="Stop")
            print("开始记录")
        else:
            self.record_button.config(text="Record")
            print("结束记录，已记录帧数:", len(self.recorded_frames))
            self.playback_recorded_frames()
            self.recorded_frames = []  # 清空录制的帧数列表
    def playback_recorded_frames(self):
        frame=np.asarray(self.recorded_frames)
        new,save_img=crop_face_resize(frames=frame, use_dynamic_detection=True,detection_freq=60, use_median_box=True, width=72, height=72)
        diff=diff_normalize_data(new)
        standardized,mean,std,L=standardized_data(new)
        light_info=[mean,std,L]
        print(light_info)
        self.mean.config(text="standard mean: {}".format(int(mean)))
        self.std.config(text="standard std： {}".format(int(std)))
        self.b.config(text="standard brightness： {}".format(int(L)))
        counter=0
        for i in range(frame.shape[0]):
            # 原始录制影像
            #frame_with_text = self.add_frame_number(self.recorded_frames[i].copy(), i + 1)
            #frame_with_text = self.add_frame_number(new[i], i + 1)
            # 亮度放大两倍的录制影像
            #enhanced_frame = self.enhance_brightness(self.recorded_frames[i], factor=2)

            # 将两个影像合并为一个图像
            #combined_frame = cv2.hconcat([frame_with_text, enhanced_frame])
            combined_frame = np.concatenate((standardized[i], diff[i]), axis=1)
            #combined_frame = np.concatenate((frame_with_text, diff[i]), axis=1)
            
            #playback_photo = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(combined_frame)))
            #self.playback_label.configure(image=playback_photo)
            #self.playback_label.image = playback_photo
            
            if np.issubdtype(combined_frame.dtype, np.floating):
                combined_frame = np.clip(combined_frame, 0, 1)  # Clip float pixel values to [0, 1]
            elif np.issubdtype(combined_frame.dtype, np.integer):
                combined_frame = np.clip(combined_frame, 0, 255)  # Clip integer pixel values to [0, 255]
            else:
                raise ValueError("Unsupported pixel value data type.")
            self.ax.clear()
            # 创建Matplotlib图形
            self.fig.tight_layout()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            self.ax.imshow(combined_frame)
            # 在Tkinter界面上显示Matplotlib图形
            self.canvas.draw()
            self.root.update_idletasks()  # 更新窗口以显示新帧
            self.root.after(2)  # 暂停一段时间以模拟视频播放速度
            counter+=15
            #if(counter>frame.shape[0]):
                #plt.savefig('test.png')
                #break
        
    def enhance_brightness(self, frame, factor):
        # 将帧转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 增加亮度
        enhanced_frame = cv2.convertScaleAbs(gray_frame, alpha=factor, beta=0)
        # 将灰度图像转换为彩色图像
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
        return enhanced_frame

    def add_timestamp(self, frame, timestamp):
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
    
    def add_frame_number(self, frame, frame_number):
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def on_close(self):
        # 关闭相机
        self.cap.release()
        self.root.destroy()

def detect(frame):
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
        
    if len(face_zone) < 1:
        print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        # Find the index of the largest face zone
        # The face zones are boxes, so the width and height are the same
        max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
        face_box_coor = face_zone[max_width_index]
        print("Warning: More than one faces are detected. Only cropping the biggest one.")
    else:
        face_box_coor = face_zone[0]
    use_larger_box=True
    larger_box_coef=1
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]    
    return face_box_coor

def standardized_data(data):
    """Z-score standardization for video data."""
    mean=np.mean(data)
    std=np.std(data)
    print(f'standardized mean：{mean} std：{std}')
    R_mean=np.mean(data[:, :, :, 0])
    G_mean=np.mean(data[:, :, :, 1])
    B_mean=np.mean(data[:, :, :, 2])
    L=0.213*R_mean+0.715*G_mean+0.072*B_mean
    print(f'Brightness = {L}')
    data = data - mean
    data = data / std
    
    data[np.isnan(data)] = 0
    return data,mean,std,L

def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data        

def crop_face_resize(frames, use_dynamic_detection, 
                         detection_freq, use_median_box, width, height):
    # Face Cropping
    if use_dynamic_detection:
        num_dynamic_det = ceil(frames.shape[0] / detection_freq)
    else:
        num_dynamic_det = 1
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    for idx in range(num_dynamic_det):
        face_region_all.append(detect(frames[detection_freq * idx]))
        print('frame ',detection_freq * idx, ' bbx: ',face_region_all[idx])
    cv2.destroyAllWindows()
    face_region_all = np.asarray(face_region_all, dtype='int')
    if use_median_box:
        # Generate a median bounding box based on all detected face regions
        face_region_median = np.median(face_region_all, axis=0).astype('int')
    print('first:',face_region_all[0])
    print('median:',face_region_median)
    
    face_region = face_region_median
    save_img=frames[0][max(face_region[1], 0):min(face_region[1] + face_region[3], frames[0].shape[0]),
                max(face_region[0], 0):min(face_region[0] + face_region[2], frames[0].shape[1])]
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
        frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frames,save_img
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraControlApp(root)
    root.mainloop()