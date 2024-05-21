import os
import pickle
import requests
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import random
import librosa
import time
from tqdm import tqdm

from utils.music_beat import get_beat

MAX_INDEX = 5000
fps = 60  # 数据集是60
movie_file_path = r"/data/datasets/AIST/video"
keypoints_file_path = r"/data/datasets/AIST/aist_plusplus_final/keypoints2d"
beats_path = r"./DATASET_ASSET/beats"
output_path = r"./DATASET_ASSET/train_data"
movie_output_path = r"./output/movie"
print(movie_file_path)
print(keypoints_file_path)


def draw_beats(beat_origin, music_beats):
    # +-70ms
    around = 15 #误差范围

    music_beats = np.array(music_beats*60, int) # 计算对应帧数

    beat_x = []
    b_idx = 0
    mb_idx = 0
    #找到距离每个音乐节拍最近的动作节拍
    for mb_idx, mb in enumerate(music_beats):
        min_ = MAX_INDEX
        min_x = -1
        while b_idx < beat_origin.shape[0]:
            sub = beat_origin[b_idx] - music_beats[mb_idx]
            if abs(sub) <= around:  # 符合要求
                if abs(sub) < min_:
                    min_ = abs(sub)
                    min_x = b_idx
                b_idx += 1
            elif sub > around:  # beat太大，进入下一个音乐beat
                break
            elif sub < -around: # beat太小，等待beat增长
                b_idx +=1
        if min_x != -1:
            beat_x.append(beat_origin[min_x])
    return beat_x

def cul_velocity(keypoints, timestamps):
    """
    计算手部点的移动速度和
    9 10
    """
    vel = np.zeros((MAX_INDEX, 17, 2))
    mask = np.zeros((MAX_INDEX))

    for i in range(1, len(keypoints)):
        if np.isnan(np.mean(keypoints[i])) or np.isnan(np.mean(keypoints[i-1])):
            mask[i] = 0
            continue
        else:
            mask[i] = 1
        for j in range(17):
            dis = keypoints[i][j] - keypoints[i-1][j]
            dis /= timestamps[i] - timestamps[i-1]
            vel[i,j] = dis[:2]
    return vel, mask

def scroll(get_frame, t):
    """
       处理每一帧图像
       """
    frame = get_frame(t)
    frame_region = frame * [1, 1, 0]
    return frame_region

def draw_beats_frame(move_beats):
    beat = np.zeros((MAX_INDEX, 1))
    for b in move_beats:
        beat[b, 0] = 1
    # pre = 0
    # for b in move_beats:
    #     if b <= pre:
    #         continue
    #     # 1 / (b - pre)
    #     for j in range(pre, b + 1):
    #         beat[j,0] = (j - pre) / (b - pre)
    #     pre = b + 1
    return beat

def vel_quantify(vel):
    """
    量化每帧的速度绝对值
    """
    vel_qtf = (vel[:,:,0] ** 2 + vel[:,:,1] ** 2) ** 0.5
    vel_qtf = np.sum(vel_qtf,axis=1)


    return vel_qtf


def normalize_keypoints(keypoints):
    """
    归一化关键点，保证上下在0-1之间
    先取头部最高点 1 2，再取脚部最低点 15 16，算出比例
    以11 12点的中点为中心归一化
    """
    mask = np.zeros(MAX_INDEX)
    
    nor_keypoints = np.zeros_like(keypoints)
    for i, key in enumerate(keypoints):
        if np.isnan(np.mean(key)):
            mask[i] = 0
            continue
        else:
            mask[i] = 1
            nor_keypoints[i,...] = keypoints[i,...]
    nor_keypoints = np.array(nor_keypoints)
    sub_keypoints = np.zeros((MAX_INDEX-nor_keypoints.shape[0],17,3))
    res_keypoints = np.concatenate((nor_keypoints, sub_keypoints))
    return res_keypoints, mask

def make_data_beats(movie_name, keypoints_path, anger):
    os.makedirs(beats_path, exist_ok=True)

    time1 = time.time()

    # keypoints 60fps
    # 读取人体关键点数据
    with open(keypoints_path, 'rb') as f:
        fr = pickle.load(f)
    keypoints = fr['keypoints2d'][anger]
    timestamps = fr['timestamps']
    timestamps = timestamps/1000000

    # 遇到nan就跳过
    if(np.isnan(np.mean(keypoints))):
        print(movie_name)
        return False

    time2 = time.time()

    # 读取音乐数据
    movie_path = os.path.join(movie_file_path, movie_name + '.mp4')
    beats_file = os.path.join(beats_path, movie_name + '.npz')
    if os.path.exists(beats_file):
        data = np.load(beats_file, allow_pickle=True)
        music_beats, music_downbeats = data['beats'], data['downbeats']
    else:
        try:
            my_clip = mp.VideoFileClip(movie_path)
        except OSError:
            url = r"https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/"
            url += movie_name + '.mp4'
            myfile = requests.get(url)
            open(movie_path,'wb').write(myfile.content)
            my_clip = mp.VideoFileClip(movie_path)
        my_clip.audio.write_audiofile(r'temp.wav')
        music_beats = get_beat(r'temp.wav')
        np.savez(beats_file, beats=music_beats)

    time3 = time.time()

    # 计算速度
    vel, mask = cul_velocity(keypoints, timestamps)
    vel_qtf = vel_quantify(vel)

    time4 = time.time()

    # 选波峰转为选波谷 向前负责，动作相比之前快，就可以认定
    vel_qtf = -vel_qtf
    beat_origin = librosa.util.peak_pick(x=vel_qtf[:keypoints.shape[0]], 
                                  pre_max=2, post_max=2,
                                  pre_avg=15, post_avg=5,
                                  wait=20, delta=1000)
    if len(beat_origin) == 0:
        print(movie_name)
        return False
    
    # 计算节奏
    beat = draw_beats(beat_origin, music_beats)
    beat_frame = draw_beats_frame(beat)

    time5 = time.time()

    #vel = vel.reshape([MAX_INDEX, 34])

    keypoints, _ = normalize_keypoints(keypoints)

    res_ts = np.zeros((MAX_INDEX - timestamps.shape[0], 1))
    timestamps = timestamps.reshape((timestamps.shape[0], 1))
    timestamps = np.concatenate((timestamps,res_ts))

    if(np.isnan(np.mean(keypoints))):
        print(movie_name)
        raise Exception("jdx nan ERROR")

    time6 = time.time()

    os.makedirs(output_path,exist_ok=True)
    output_file = os.path.join(output_path, movie_name + '.npz')
    np.savez(output_file, keypoints=keypoints, timestamps=timestamps, beat=beat_frame, mask=mask)

    time7 = time.time()

    # print("all={:.2f},12={:.2f},23={:.2f},34={:.2f},45={:.2f},56={:.2f},67={:.2f}".format(time7-time1, time2-time1,time3-time2,time4-time3,time5-time4,time6-time5,time7-time6),end='\r')

    # 画个图算delta
    # v_x = []
    # v_y = []
    # for i, v in enumerate(-vel_qtf):
    #     v_x.append(i)
    #     v_y.append(v)
    # plt.plot(v_x[60:150], v_y[60:150], color='b')
    # 画节拍
    # for b in beat[1:3]:
    #     x = [b, b]
    #     y = [0, 9000]
    #     plt.plot(x, y, color='g')

    # x = []
    # y = []
    # beat = [120,830]
    # for i in range(0,1000):
    #     x.append(i)
    #     if(i in beat):
    #         y.append(1)
    #     else:
    #         y.append(0)
    # plt.plot(x, y, color='g')

    # plt.savefig('power.jpg')



    # # 加特技
    # pre = 0
    # my_clip = mp.VideoFileClip(movie_path)
    # outputClip = my_clip.subclip(0.01, 0.02)
    # for idx in beat:
    #     beat_time = idx/60
    #     if beat_time < 0.1:  # 一开始就节奏不好看
    #         continue
    #     clip1 = my_clip.subclip(pre, beat_time - 0.05)
    #     clip2 = my_clip.subclip(beat_time - 0.05, beat_time + 0.05)
    #     clip2_modi = clip2.fl(scroll)
    #     outputClip = mp.concatenate_videoclips([outputClip, clip1, clip2_modi])
    #     pre = beat_time + 0.05
    # if pre < my_clip.end:
    #     clip1 = my_clip.subclip(pre)
    #     outputClip = mp.concatenate_videoclips([outputClip, clip1]) 
    # os.makedirs(movie_output_path, exist_ok=True)
    # movie_output_file = os.path.join(movie_output_path, movie_name+'.mp4')
    # outputClip.write_videofile(movie_output_file)

    return True


def main():
    movie_path_list = os.listdir(movie_file_path)
    movie_name_list = []

    for movie_path in tqdm(movie_path_list):
    # for movie_path in movie_path_list:
        # print("Start {}".format(movie_path))
        movie_name = movie_path.split('.')[0]
        flag = True
        if not os.path.exists(os.path.join(output_path, movie_name + '.npz')):
            name_list = movie_name.split('_')
            anger = int(name_list[2][1:]) - 1
            name_list[2] = "cAll"
            keypoints_path = os.path.join(keypoints_file_path, '_'.join(name_list) + '.pkl')
            flag = make_data_beats(movie_name, keypoints_path, anger)

        if flag:
            movie_name_list.append(movie_name)

    length = len(movie_name_list)
    train_length = round(length * 0.9)
    random.shuffle(movie_name_list)
    with open('DATASET_ASSET/test_data.csv', 'w', encoding='utf8') as f:
        f.writelines([line+'\n' for line in movie_name_list[train_length:]])

    with open('DATASET_ASSET/train_data.csv', 'w', encoding='utf8') as f:
        f.writelines([line+'\n' for line in movie_name_list[:train_length]])


if __name__ == "__main__":


    main()

    # make_data_beats("gBR_sBM_c01_d04_mBR0_ch05",
    #                 "/data/datasets/AIST/aist_plusplus_final/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch05.pkl",
    #                 0)
    # https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/
    # gBR_sBM_c01_d04_mBR0_ch05
    # gLH_sBM_c01_d16_mLH3_ch07
    # gLO_sBM_c01_d14_mLO4_ch04

    # nan gWA_sBM_c07_d25_mWA2_ch09
