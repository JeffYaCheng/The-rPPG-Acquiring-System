import os
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin,filtfilt

def read_wave(path:str,Filter=False,type='ECG'):
    """Samples a PPG sequence into specific length."""
    signal=pd.read_csv(path)
    signal=signal.values.reshape(1,-1).squeeze()
    
    if(Filter):
        if(type=='ECG'):
            fs=1000
            taps = firwin(300, [0.5 / (fs / 2), 55 / (fs / 2)], pass_zero=False)
            signal =filtfilt(taps,1, np.double(signal))
        else:
            #remove PPG outlier
            #print('shape=',signal.shape[0])
            #plt.title('original PPG')
            #plt.plot(signal)
            #plt.show()
            mean=np.mean(signal) #be used to reduce peak
            std=np.std(signal)
            outlier=signal<(mean+5*std)
            for i in range(len(outlier)):
                if(outlier[i]==False):
                    #print('========================outlier 1 ===========================')
                    signal[i]=signal[i-1]
            outlier=signal>(mean-5*std)
            for i in range(len(outlier)):
                if(outlier[i]==False):
                    #print('========================outlier 2 ===========================')
                    signal[i]=signal[i-1]
            #print('shape=',signal.shape[0])
            #plt.title('remove outlier PPG')
            #plt.plot(signal)
            #plt.show()
            fs=100
            taps = firwin(300, [0.5 / (fs / 2), 5 / (fs / 2)], pass_zero=False)
            signal =filtfilt(taps,1, np.double(signal))
        #print(signal.shape)
    signal=signal.tolist()
    signal=signal
    for j in range(len(signal)):
        signal[j]=int(signal[j])
    input_signal=np.asarray(signal)
    return len(signal),input_signal

def clip_normalize(signal,fps,signal_time,signal_range,img_counter,img_time,type='ECG'):
    #print('ori signal=',signal.shape)
    counter_procss_n_img=0
    delete_img=[]
    normalize_signal=None
    resample_len=[]
    save_time=[]
    img_len=[]
    first=True
    mean=np.mean(signal) #be used to reduce peak
    std=np.std(signal)
    total_img=0
    for i in range(len(img_time)):
        total_img+=img_counter[i]
        t_img_time=img_time[i]
        have_signal=False
        #find the signal that have the same time with img
        for j in range(len(signal_time)):
            if(signal_time[j]==t_img_time):
                have_signal=True
                break
        if(have_signal):
            if(type=='ECG' or type=='PPG'):
                #target_length=int(img_counter[i]*fps/30)
                save_time.append(float(t_img_time)/10)
                img_len.append(img_counter[i])
                #find original signal interval
                c_signal_range=signal_range[j]
                begin=c_signal_range[0]
                end=c_signal_range[1]

                input_signal=signal[begin:end]
                target_len=img_counter[i]*20 # len/30*600
                delete_signal=input_signal.shape[0]-target_len
                #print(' time=',float(t_img_time)/10,' input_signal=',input_signal.shape[0],' img_counter=',img_counter[i],' delete=',delete_signal)
                if(delete_signal>=0):
                    gap=input_signal.shape[0]/target_len
                    left_point=list()
                    for m in range(0,target_len,1):
                        point=int(gap*m)
                        left_point.append(point)
                    left=[]
                    for m in range(0,input_signal.shape[0]):
                        if m in left_point:
                            left.append(True)
                        else:
                            left.append(False)
                    n_signal=input_signal[left]
                    #if(n_signal.shape[0]!=target_len):
                        #print('resample shape=',n_signal.shape[0])
                        #print(' time=',float(t_img_time)/10,' input_signal=',input_signal.shape[0],' img_counter=',img_counter[i],'target len=',target_len,' delete=',delete_signal)
                        #print(left)
                else:
                    #print('==========================ECG signal is not enough=============================')
                    #print(' time=',float(t_img_time)/10,' input_signal=',input_signal.shape[0],' img_counter=',img_counter[i],'target len=',target_len,' delete=',delete_signal)
                    n_signal=np.interp(np.linspace(1, input_signal.shape[0], target_len), 
                                       np.linspace(1, input_signal.shape[0], input_signal.shape[0]), 
                                       input_signal)
                    #print('resample shape=',n_signal.shape[0])
                resample_len.append(n_signal.shape[0])

                #合併resample的訊號
                if first:
                    normalize_signal=n_signal
                    first=False
                else:
                    normalize_signal=np.concatenate((normalize_signal,n_signal))
                counter_procss_n_img+=img_counter[i]
            elif(type=='rPPG'):
                save_time.append(float(t_img_time)/10)
                img_len.append(img_counter[i])
                #find original signal interval
                c_signal_range=signal_range[j]
                begin=c_signal_range[0]
                end=c_signal_range[1]

                input_signal=signal[begin:end]
                
                delete_signal=input_signal.shape[0]-img_counter[i]
                #print(' time=',float(t_img_time)/10,' input_signal=',input_signal.shape[0],' img_counter=',img_counter[i],' delete=',delete_signal)
                if(delete_signal>=0):
                    gap=input_signal.shape[0]/img_counter[i]
                    left_point=list()
                    for m in range(0,img_counter[i],1):
                        point=int(gap*m)
                        left_point.append(point)
                    left=[]
                    for m in range(0,input_signal.shape[0]):
                        if m in left_point:
                            left.append(True)
                        else:
                            left.append(False)
                    n_signal=input_signal[left]
                    #if(n_signal.shape[0]!=img_counter[i]):
                        #print('resample shape=',n_signal.shape[0])
                        #print(' time=',float(t_img_time)/10,' input_signal=',input_signal.shape[0],' img_counter=',img_counter[i],' delete=',delete_signal)
                        #print(left)
                else:
                    #print('==========================PPG signal is not enough=============================')
                    #print(' time=',float(t_img_time)/10,' input_signal=',input_signal.shape[0],' img_counter=',img_counter[i],'target len=',img_counter[i],' delete=',delete_signal)
                    n_signal=np.interp(np.linspace(1, input_signal.shape[0], img_counter[i]), 
                                       np.linspace(1, input_signal.shape[0], input_signal.shape[0]), 
                                       input_signal)
                    #print('resample shape=',n_signal.shape[0])
                resample_len.append(n_signal.shape[0])
                #合併resample的訊號
                if first:
                    normalize_signal=n_signal
                    first=False
                else:
                    normalize_signal=np.concatenate((normalize_signal,n_signal))
                counter_procss_n_img+=img_counter[i]
                
        else: #相片對應不到訊號
            start_del=counter_procss_n_img+1
            counter_procss_n_img+=img_counter[i]
            for i in range(start_del,counter_procss_n_img+1):
                delete_img.append(i)
    return normalize_signal,delete_img,resample_len,save_time,img_len
def normalize(root,file_dir):
    
    normalize_dir=os.path.join(root, os.path.basename(file_dir))
    if not os.path.exists(normalize_dir):
        os.makedirs(normalize_dir)
    else:
        print('this file has been normalize')
        return 

    ppgr_path=f'{file_dir}/PPG_R.csv'
    ppgir_path=f'{file_dir}/PPG_IR.csv'
    ecg_path=f'{file_dir}/ECG.csv'
    img_path=f'{file_dir}/img/*.png'
    record_info=f'{file_dir}/img/time_step.txt'
    ppg_time_step=f'{file_dir}/ECG_PPG_time_step.txt'
    img=glob.glob(img_path)
    
    # copy img
    source_folder = os.path.join(file_dir, 'img')
    destination_folder = os.path.join(normalize_dir, 'img')
    #print('destination_folder=',destination_folder)
    #print('source_folder=',source_folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        # fetch all files
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = os.path.join(source_folder, file_name)
            destination = os.path.join(destination_folder, file_name)
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
                #print('copied', file_name)
    else:
        print('this file has been normalize')
        return 
        #shutil.rmtree(destination_folder)
        #os.makedirs(destination_folder)
    
    # copy original signal
    destination_folder = os.path.join(normalize_dir, 'raw_data')
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    shutil.copy(ppgr_path, destination_folder)
    shutil.copy(ppgir_path, destination_folder)
    shutil.copy(ecg_path, destination_folder)
    shutil.copy(ppg_time_step, destination_folder)
    #print(ppgir_path,'  copy to ',normalize_dir)
    
    #read ppg ecg timestep
    ppg_time=[] #12,13,14,15s
    ppg_counter=[] #[9,20],[20,68] 第12秒時，對應的訊號為9-20格
    ecg_counter=[]
    with open(ppg_time_step) as f:
        last=0
        ecg_end=0
        ppg_end=0
        for line in f.readlines():
            s = line.split(',') # timestep,ECG此次封包的點數,ppg此次封包的點數
            time=int(float(s[0])*10) #0.1s 同步一次
            if last==time:
                ecg_end+=int(s[1])
                ppg_end+=int(s[2])
            else:
                if(last!=0):
                    ppg_time.append(last)
                    ppg_counter.append([ppg_start,ppg_end])
                    ecg_counter.append([ecg_start,ecg_end])
                    record=True
                ecg_start=ecg_end
                ppg_start=ppg_end
                ecg_end+= int(s[1])
                ppg_end+=int(s[2])
                last=time        
        ppg_time.append(last)
        ppg_counter.append([ppg_start,ppg_end])
        ecg_counter.append([ecg_start,ecg_end])      
                
    
    #get time info
    img_time=list()#12,13,14,15s
    img_counter=list() #3,4,5 第12秒時，有3張照片
    last=None
    #img_time_step=list() 
    with open(record_info, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        record_time=str1[0]
        counter=0 #紀錄現在處理到第幾張照片，讀
        time_counter=0 #記錄每0.1秒有幾張照片
        for time in str1[1].split():
            x=time
            if(counter==0):
                x=x[1:-1] #去[,
                last=int(float(x)*10) #record last time 1.1s 1.1s 1.1s => 11s 11s 11s
                time_counter=0
            elif(counter==len(str1[1].split())-1):
                #print('counter=',counter)
                x=x[:-1] #去]
                #print('time lengh=',len(str1[1].split()))
            else:
                x=x[:-1] #去,
                
            #print('int=',last,'x=',x)
            
            
            if(last==int(float(x)*10)):
                time_counter+=1
                    
            else:
                img_time.append(last)
                img_counter.append(time_counter)
                time_counter=1
                last=int(float(x)*10)
            counter+=1
        #print('last img time = ',last)
        img_time.append(last)
        img_counter.append(time_counter)
    
    
    #read wave
    ecg_len,ecg=read_wave(ecg_path,Filter=True,type='ECG')
    ppgr_len,ppgr=read_wave(ppgr_path,Filter=True,type='PPG')
    ppgir_len,ppgir=read_wave(ppgir_path,Filter=True,type='PPG')
    # normalize
    re_ecg,delete_img,resample_len_ecg,save_time,img_len=clip_normalize(ecg,1000,ppg_time,ecg_counter,img_counter,img_time,'ECG')
    re_ppgr,delete_img,resample_len_ppgr,save_time,img_len=clip_normalize(ppgr,100,ppg_time,ppg_counter,img_counter,img_time,'rPPG')
    re_ppgir,delete_img,resample_len_ppgir,save_time,img_len=clip_normalize(ppgir,100,ppg_time,ppg_counter,img_counter,img_time,'rPPG')
    

    #delete img that can't get signal
    for i in delete_img:
        counter=f'{i:05d}.png'
        img_name=f'{normalize_dir}/img/{counter}'
        print('==========================================delete img =============================================')
        #print(img_name)
        #img_num-=1
        if os.path.isfile(img_name):
            print(img_name)
            os.remove(img_name)
        
    
    #rename img that can be transformed to video
    path=f'{normalize_dir}/img'
    img_list=os.listdir(path)
    img_list.remove('time_step.txt')
    img_list.sort(key=lambda x :int(x[:-4]))
    n = 1          # 設定名稱從 1 開始
    for i in img_list:
        name=os.path.join(path,f'{n:05d}.png')
        os.rename(os.path.join(path,i), name)   # 改名時，使用字串格式化的方式進行三位數補零
        n = n + 1    # 每次重複時將 n 增加 1
    
    time_step= open(os.path.join(normalize_dir, 'syn_time_step.txt'), "w")
    for i in range(len(img_len)):
        time_step.write(f'{save_time[i]} {resample_len_ecg[i]} {resample_len_ppgir[i]} {resample_len_ppgr[i]} {img_len[i]}\n')

    df = pd.DataFrame(re_ppgr)
    df.to_csv(os.path.join(normalize_dir, 'syn_PPG_R.csv'), index=False)
    df = pd.DataFrame(re_ppgir)
    df.to_csv(os.path.join(normalize_dir, 'syn_PPG_IR.csv'), index=False)
    df = pd.DataFrame(re_ecg)
    df.to_csv(os.path.join(normalize_dir, 'syn_ECG.csv'), index=False)
    

    FileName = f'syn_information.txt'
    information= open(os.path.join(normalize_dir, FileName), "w")
    information.write(f'ori name ={os.path.basename(file_dir)}\n')
    information.write(f'actually record time = {record_time}\n')
    information.write(f'calculate record video time  =  {len(img)/30}\n')
    information.write(f'calculate record ECG time  =  {ecg_len/1000}\n')
    information.write(f'original ECG length    = {ecg_len} normalize length = {re_ecg.shape[0]}\n')
    information.write(f'original PPG_R length  =  {ppgr_len} normalize length = {re_ppgr.shape[0]}\n')
    information.write(f'original PPG_IR length =  {ppgir_len} normalize length = {re_ppgir.shape[0]}\n')
    information.write(f'original image length  =  {len(img)} normalize length = {len(img_list)}\n')
    
    information.close()

    cmd=f'ffmpeg -r 30 -i {normalize_dir}/img/%05d.png -an -c:v rawvideo -pix_fmt bgr24 {normalize_dir}/output.avi'
    res=os.popen(cmd)
    print('Normalize complete')

import time
import argparse
parser=argparse.ArgumentParser()
TriAnswer_PROG_ROOT = os.path.abspath(os.getcwd())
TriAnswer_Record_Dir = os.path.join(TriAnswer_PROG_ROOT, 'TriAnswer_Records')
Normalize_Dir = os.path.join(TriAnswer_PROG_ROOT, 'Normalize')
dirIsExist = os.path.exists(Normalize_Dir)
if not dirIsExist:
    os.makedirs(Normalize_Dir)

parser.add_argument('-i','--input_file_path',type=str,default=TriAnswer_Record_Dir)
parser.add_argument('-o','--output_file_path',type=str,default=Normalize_Dir)
args=parser.parse_args()

file_path=args.input_file_path
save_path=args.output_file_path
data_dir=os.listdir(file_path)
for i in range(len(data_dir)):
    print(os.path.join(file_path, data_dir[i]),'  =>  ',save_path,data_dir[i])
    normalize(save_path,os.path.join(file_path, data_dir[i]))
    #time.sleep(30)