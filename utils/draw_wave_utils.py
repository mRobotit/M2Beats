import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def cul_velocity(keypoints, fps):
    """
    计算手部点的移动速度和
    9 10
    """
    vel = np.zeros((len(keypoints), 17, 2))
    pre = np.array([0,0])
    for i in range(1, len(keypoints)):
        for j in range(17):
            dis = keypoints[i][j] - keypoints[i-1][j]
            dis *= fps
            vel[i,j] = (dis[:2] + pre) / 2
            pre = dis[:2]
    return vel

def vel_quantify(vel):
    """
    量化每帧的速度绝对值
    """
    vel_qtf = (vel[:,:,0] ** 2 + vel[:,:,1] ** 2) ** 0.5
    vel_qtf = np.sum(vel_qtf,axis=1)
    return vel_qtf

def draw_wave(asset_path, movie_path, fps, keypoints_path, music_beat):
    """
    params
    asset_path: 目录位置
    movie_path: 视频文件位置
    fps: 视频帧率
    keypoints_path: 关键点位置（画图用）
    music_beat: 音乐节奏（画图用）
    """
    row_fig_jpg_path = os.path.join(asset_path, "row_fig_jpg")
    os.makedirs(row_fig_jpg_path, exist_ok=True)
    row_fig_movie_path = os.path.join(asset_path, "row_fig_movie")
    os.makedirs(row_fig_movie_path, exist_ok=True)

    fr = np.load(keypoints_path, allow_pickle=True)
    keypoints = fr['keypoints']
    vel = cul_velocity(keypoints, fps)
    vel_qtf = vel_quantify(vel)

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot()

    frame_start = 0
    # frame_num = len(vel_qtf)
    frame_num = 120

    for i in tqdm(range(frame_start, frame_num)):
        ax.set_xlim(frame_start/fps, frame_num/fps)
        ax.set_ylim(0,9000)
        ax.autoscale(False)
        ax.scatter([1024/fps],[90000], color='b')
        v_x = []
        for j in range(frame_start,i):
            v_x.append(j/fps)
        ax.plot(v_x,vel_qtf[frame_start:i], color='b')
        
        v_x = []
        v_y = []
        for mc in music_beat:
            if mc > i/fps:
                break
            elif mc > frame_start / fps:
                v_x = [mc, mc]
                v_y = [0000, 8000]
                ax.plot(v_x,v_y, color='g')
        fig_path = os.path.join(row_fig_jpg_path, "{}.jpg".format(i))
        fig.savefig(fig_path)
        ax.cla()
    fig_movie_path = os.path.join(row_fig_movie_path, "demo.avi")
    size = (1000,200)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    caw = cv2.VideoWriter(fig_movie_path,fourcc,fps,size)
    for i in range(frame_num):
        filename = os.path.join(row_fig_jpg_path, "{}.jpg".format(i))
        img = cv2.imread(filename)
        if img is None:
            print(filename + "为空!")
            continue
        caw.write(img)
    caw.release()