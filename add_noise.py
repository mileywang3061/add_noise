# -*- coding:utf-8 -*-
import numpy as np
import librosa
import os
import scipy.io.wavfile as wavfile
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as newPool
import math
import logging
import multiprocessing

# import time

# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(filename='test.log', level=logging.INFO, format=LOG_FORMAT)
# from multiprocessing import Process
# import multiprocessing as mp
# 无噪文件路径
# filepath ="C:/Users/ubt/Desktop/S0724"
filepath = "/data/xianjieyang/data/audio/data_aishell/wav"
# 噪声路径
# noisepath ="E:/adding noise/noise_sub"
noisepath = "/data/mileywang/noisewav"
# 存储路径
savepath = '/data/mileywang/noise_combination/'
# savepath='E:/adding noise/new_combination'
txtpath = '/data/mileywang/txt/'
# txtpath='C:/Users/ubt/Desktop/BAC009S0916W0490.txt'
# 噪声切片
def cut_noise(wav_file, noise_file):
    # 随机截取噪声音频中的片段(要求和无噪音频长度相同)
    start_time = np.random.uniform(len(noise_file) - len(wav_file))
    end_time = start_time + len(wav_file)
    start_time = int(start_time)
    end_time = int(end_time)
    bg_slice = noise_file[start_time: end_time]
    return bg_slice

# 经过VAD检测后加入噪声
def add_noise(wav_file, cutnoise_file, SNR,wav_file_name,txt_file_name):
    Noise = cutnoise_file - np.mean(cutnoise_file)
    signal_power = (1 / len(wav_file)) * sum(wav_file * wav_file)
    noise_variance = signal_power / (pow(10, (SNR / 10)))
    noise = np.sqrt(noise_variance) / np.std(Noise) * Noise
    # print(noise)
    os.system('/data/xianjieyang/code/speech-vad-rtc/build/vad-demo %s %s' % (wav_file_name, txt_file_name))
    label = open(txt_file_name, 'r')
    data = label.readlines()
    head_diff = int(data[0])
    head_length = len(wav_file_name) - head_diff
    head_length = int(head_length)
    # print(head_length)
    del data[0]
    new_label = []
    for i in data:
        new_label.append(int(i[-2]))
    # print(new_label)
    # print(new_label)
    # 加噪
    # use VAD to add part of the pieces
    with_noise=[]
    for i in range(0, len(new_label)):
        # 音频头部 信息不发生改变
        if i == 0:
            with_noise.append((wav_file[int(0):int(head_length)]))
            # print(with_noise)
        else:
            with_noise.append((wav_file[int((i * 160)+ head_length):int(((i+1) * 160)+ head_length)] + noise[int((i * 160)+ head_length):int(((i+1) * 160)+ head_length)]))            # print(with_noise)
        # print(with_noise)
    with_noise = np.array(with_noise)
    with_noise = with_noise.flatten()
    # print(len(with_noise))
    with_noise = np.concatenate(with_noise, axis=0)
    # print(len(with_noise))
    # print(with_noise)
    return with_noise


    # with_noise = np.array(with_noise)
    # with_noise = with_noise.flatten()
    # with_noise = np.concatenate(with_noise, axis=0)
# voice,sr = librosa.load('C:/Users/ubt/Desktop/S0724/BAC009S0724W0121.wav', sr=None)
# noise ,sr = librosa.load('E:/adding noise/noise_sub/babble.wav', sr=None)
# noise_file=cut_noise(voice,noise)
# print(noise_file)
# txtfile ='C:/Users/ubt/Desktop/BAC009S0916W0490.txt'
# with_noise=add_noise(voice,noise_file,0, ,txtfile)
# print(with_noise)

# 多进程并行化
def parallel_file(i, poolSize):
    # poolSize=6
    pathList = []
    cycle = 0
    # print(filepath + str(i))
    for root, directory, files in os.walk(filepath):
        folder = os.path.relpath(root, filepath)
        if len(directory) == 0 and len(files) > 0:
            # print(root + " " +  str(i) + " " + str(len(files)))
            cycle = int(math.ceil(len(files) / poolSize))
            for x in range(cycle):
                if i * cycle + x < len(files):
                    file_name = os.path.join(root, files[i * cycle + x])
                        # 判断音频文件是否为空
                    t = os.path.getsize(file_name)
                    # print(t)
                    if t < (5 * 1024):
                        print(file_name)
                    else:
                        voice_wav, sr = librosa.load(file_name, sr=None)
                        # print(voice_wav)
                        txt_file = txtpath + os.path.basename(file_name).strip(".wav") + '.txt'
                        # print(voice_wav)
                            # raw_file = rawpath + os.path.basename(file_name).strip(".wav") + '.raw'
                            # raw_file = file_name.strip(".wav")+'.raw'
                            # txt_file = file_name.strip(".wav")+'.txt'
                            # txt_file = txtpath + os.path.basename(file_name).strip(".wav") + '.txt'
                        for noise_root, noise_directory, noise_files in os.walk(noisepath):
                            for noise_file in noise_files:
                                    # 噪声文件夹名称
                                noise_file_name = os.path.join(noise_root, noise_file)
                                noise_wav, sr = librosa.load(noise_file_name, sr=None)
                                noise_part = cut_noise(voice_wav, noise_wav)
                                    # 加入生成信噪比（信噪比由自己决定）的合成噪音
                                with_noise = add_noise(voice_wav, noise_part, 0,file_name, txt_file)
                                # print(with_noise)
                                new_savepath = savepath +'/'+ os.path.basename(noise_file_name).strip(
                                        ".wav") + "//" + folder
                                if not os.path.exists(new_savepath):
                                    os.makedirs(new_savepath)
                                    # 保存的文件名
                                new_save_name = os.path.join(
                                    new_savepath + "//" + os.path.basename(file_name).strip(".wav") +'.'+ os.path.basename(
                                        noise_file_name))
                                with_noise = (with_noise) * 32767
                                for z in range(len(with_noise)):
                                    if with_noise[z] > 32767:
                                        with_noise[z] = 32767
                                    elif with_noise[z] < (-32767):
                                        with_noise[z] = -32767
                                wavfile.write(new_save_name, sr, with_noise.astype(np.int16))
                    pathList.append(os.path.join(root, files[i * cycle + x]))


# 多进程并行化变速不变调(Soundtouch)
# def change_tempo(i, poolSize):
#     for root, directory, files in os.walk(filepath):
#         folder = os.path.relpath(root, filepath)
#         if len(directory) == 0 and len(files) > 0:
#             # print(root + " " +  str(i) + " " + str(len(files)))
#             cycle = int(math.ceil(len(files) / poolSize))
#             print(cycle)
#             for x in range(cycle):
#                 if i * cycle + x < len(files):
#                     file_name = os.path.join(root, files[i * cycle + x])
#                     # 判断音频文件是否为空
#                     t = os.path.getsize(file_name)
#                     # print(t)
#                     if t < (5 * 1024):
#                         print(file_name)
#                     else:
#                         tempo_file = ratepath + "//" + folder
#                         if not os.path.exists(tempo_file):
#                             os.makedirs(tempo_file)
#                         tempofile = os.path.join(tempo_file + "//" +os.path.basename(file_name).strip(".wav")+'tempo'+".wav")
#                         os.system('soundstretch %s %s -tempo=15' % (file_name, tempofile))

# def change_pitch(i,poolSize):
#     for root, directory, files in os.walk(filepath):
#         folder = os.path.relpath(root, filepath)
#         if len(directory) == 0 and len(files) > 0:
#             # print(root + " " +  str(i) + " " + str(len(files)))
#             cycle = int(math.ceil(len(files) / poolSize))
#             print(cycle)
#             for x in range(cycle):
#                 if i * cycle + x < len(files):
#                     file_name = os.path.join(root, files[i * cycle + x])
#                     # 判断音频文件是否为空
#                     t = os.path.getsize(file_name)
#                     # print(t)
#                     if t < (5 * 1024):
#                         print(file_name)
#                     else:
#                         pitch_file = pitchpath + "//" + folder
#                         if not os.path.exists(pitch_file):
#                             os.makedirs(pitch_file)
#                         pitchfile = os.path.join(pitch_file + "//" + os.path.basename(file_name))
#                         os.system('soundstretch %s %s -pitch=15' % (file_name, pitchfile))

# 主方法
if __name__ == '__main__':
    # poolSize为进程数 可自行定义（进程数）
    poolSize = 10
    p = Pool(poolSize)
    # print("start ....")
    for i in range(poolSize):
        num = multiprocessing.Value("d", i)
        # print("threading:" + str(i))
        logging.info("进程：" + str(i) + "开始")
        # p.apply_async(parallel_file, args=(i, poolSize,))
        p.apply_async(parallel_file, args=(i, poolSize,))
        # p.apply_async(change_pitch, args=(i, poolSize,))
    p.close()
    p.join()
