import numpy as np

def movie_two_music(movie_beat_list, music_beat_list, TIME_UP=True):
    # 为所有 动作节奏 匹配 音乐节奏
    # 如果有多个 动作节奏 的最近都指向同一个 音乐节奏，那么只有一个动作节奏生效，其他失效
    two_list = []

    # 最左匹配 LEFT_UP 时间最优 TIME_UP
    if TIME_UP:
        mus_i = 1
        sub = 0
        for i, movie_beat in enumerate(movie_beat_list):
            new_movie_beat = movie_beat + sub
            # 保证mus_i时间上比动作节奏大 mus_i-1 比小
            while music_beat_list[mus_i] < new_movie_beat:
                mus_i += 1
                if mus_i == len(music_beat_list):
                    break
            if mus_i == len(music_beat_list):
                break
            
            dis1 = new_movie_beat - music_beat_list[mus_i - 1]
            dis2 = music_beat_list[mus_i] - new_movie_beat
            if dis1 < dis2:
                # 音乐节奏不能为0
                if mus_i-1 == 0:
                    continue
                # 匹配左侧的节奏
                if len(two_list) == 0:
                    two_list.append([i,mus_i-1,dis1])
                elif two_list[-1][1] == mus_i-1 and two_list[-1][2] > dis1:
                    two_list[-1] = [i,mus_i-1,dis1]
                elif two_list[-1][1] != mus_i-1:
                    two_list.append([i,mus_i-1,dis1])
                sub = sub - dis1
            else:
                # 音乐节奏不能为0
                if mus_i == 0:
                    continue
                # 匹配右侧的节奏
                if len(two_list) == 0:
                    two_list.append([i,mus_i,dis1])
                elif two_list[-1][1] == mus_i and two_list[-1][2] > dis2:
                    two_list[-1] = [i,mus_i,dis2]
                elif two_list[-1][1] != mus_i:
                    two_list.append([i,mus_i,dis2])
                sub = sub + dis2
    else:
        i=0
        while(i<len(movie_beat_list) and i<len(music_beat_list)):
            two_list.append([i,i+1])
            i+=1
    return two_list

def two2acc(two_list, movie_beat_list, music_beat_list):
    acc_list = [] # [start, end, target_time]
    pre_movie = 0
    pre_music = 0
    for two in two_list:
        movie_idx = two[0]
        music_idx = two[1]
        acc_list.append([pre_movie, movie_beat_list[movie_idx], music_beat_list[music_idx] - pre_music])
        pre_movie = movie_beat_list[movie_idx]
        pre_music = music_beat_list[music_idx]
    return acc_list

def boom_movie_beat(movie_beat_list, music_beat_list, fps):
    assert type(movie_beat_list) is np.ndarray
    assert type(music_beat_list) is np.ndarray

    # 如果是帧数表示，转换为时间
    if type(movie_beat_list[0]) in [int,np.int32,np.int64]:
        movie_beat_list = movie_beat_list / fps
    
    # 第一个音乐节奏设置为0 方便编码
    music_beat_list = np.insert(music_beat_list, 0, 0)

    two_list = movie_two_music(movie_beat_list, music_beat_list)

    acc_list = two2acc(two_list, movie_beat_list, music_beat_list)

    return acc_list
    
    
if __name__ == '__main__':
    movie_beat_list = np.array([30,60,90,120,150])
    music_beat_list = np.array([0.5,2,3,4,5])
    boom_movie_beat(movie_beat_list, music_beat_list, 30)