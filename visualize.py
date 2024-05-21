import os
import numpy as np
import moviepy.editor as mp


def scroll(get_frame, t):
    """
       处理每一帧图像
       """
    frame = get_frame(t)
    frame_region = frame * [1, 1, 0]
    return frame_region

def beat_on_movie(movie_path, beat_list, fps):
    # 加特技
    pre = 0
    my_clip = mp.VideoFileClip(movie_path)
    outputClip = my_clip.subclip(0.01, 0.02)
    for idx in beat_list:
        beat_time = idx/fps
        if beat_time < 0.1:  # 一开始就节奏不好看
            continue
        clip1 = my_clip.subclip(pre, beat_time - 0.1)
        clip2 = my_clip.subclip(beat_time - 0.1, beat_time + 0.1)
        clip2_modi = clip2.fl(scroll)
        outputClip = mp.concatenate_videoclips([outputClip, clip1, clip2_modi])
        pre = beat_time + 0.1
    if pre < my_clip.end:
        clip1 = my_clip.subclip(pre)
        outputClip = mp.concatenate_videoclips([outputClip, clip1]) 

    os.makedirs("./output/movie",exist_ok=True)
    movie_out_path = os.path.join("./output/movie",os.path.basename(movie_path))
    outputClip.write_videofile(movie_out_path)

def main():
    movie_name = "gBR_sBM_c01_d05_mBR0_ch08"
    print("visualize {}".format(movie_name))
    movie_path = os.path.join(movie_file_path,movie_name+ ".mp4") 

    input_file = os.path.join(note_file_dir, movie_name + '.npz')
    fr = np.load(input_file, allow_pickle=True)
    keypoints = fr['keypoints'][:1024] # 9999,34
    beat = fr['beat'] # 5000,1   beat
    mask = fr['mask']
    length = np.where(mask==1)[0][-1]
    beat = beat[:length]
    beat_pos = np.where(beat==1)[0]
    print("beat_pos: {}".format(beat_pos))

    human_pos = np.array([49, 82, 104, 139, 182, 224, 260, 283, 315, 362, 402, 437, 458, 498, 543, 586, 623, 643, 673])

    beat_on_movie(movie_path, beat_pos, 60)

def ret_60_100(human_list_origin):
    human_pos = []
    for h in human_list_origin:
        min = h[0]
        sec = h[1]
        min_sec = min + sec/60

        idx = min_sec * 60
        human_pos.append(idx)
    human_pos = np.array(human_pos, int)
    return human_pos


if __name__=="__main__":
    movie_file_path = r"/data/datasets/AIST/video"
    note_file_dir = r"/data/jdx/code/mycode/train_data"
    main()