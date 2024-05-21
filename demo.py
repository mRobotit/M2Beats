import os
import moviepy.editor
import numpy as np

from utils.wild_movie_keypoints import movie_keypoints
from utils.inf_video_wild import inference
from utils.music_beat import get_beat
from utils.movie_acc_arrange import boom_movie_beat
from utils.movie_acc_utils import change_speed
from utils.draw_wave_utils import draw_wave

movie_path = "test_movie/gBR_sBM_c01_d05_mBR0_ch08.mp4"
music_path = "./test_movie/believer.mp4"
M2B_checkpoint = "checkpoint/loss_40_test_42_params.pt"

asset_path = r"./MOVIE_ASSET/"

def main():
    movie_name = os.path.basename(movie_path).split('.')[0]
    music_name = os.path.basename(music_path).split('.')[0]
    video = moviepy.editor.VideoFileClip(movie_path)
    fps = video.fps

    # get keypoints
    print("--------------------GET KEYPOINTS--------------------------")
    keypoints_output_path = os.path.join(asset_path, "keypoints")
    os.makedirs(keypoints_output_path, exist_ok=True)
    movie_keypoints(movie_path, keypoints_output_path)

    # get movie beat
    print("--------------------GET MOVIE BEAT--------------------------")
    mvbeat_output_path = os.path.join(asset_path, "mvbeat")
    os.makedirs(mvbeat_output_path, exist_ok=True)
    keypoints_path = os.path.join(keypoints_output_path, "{}.npz".format(movie_name))
    inference(keypoints_path, movie_name, M2B_checkpoint, fps, mvbeat_output_path)

    # get music beat
    print("--------------------GET MUSIC BEAT--------------------------")
    if(os.path.basename(music_path).split('.')[-1] in ['mp4','avi']):
        my_clip = moviepy.editor.VideoFileClip(music_path)
        music_output_path = os.path.join(asset_path, "music")
        os.makedirs(music_output_path, exist_ok=True)
        music_asset_path = os.path.join(music_output_path, "{}.wav".format(music_name))
        my_clip.audio.write_audiofile(music_asset_path)
    else:
        music_asset_path = music_path
    music_beat = get_beat(music_asset_path)


    # generate old demo
    print("-----------------------GENERATE OLD DEMO------------------------------")
    draw_wave(asset_path, movie_path, fps, keypoints_path, music_beat)

    # arrage movie and music beat
    print("--------------------ARRANGE--------------------------")
    mvbeat_path = os.path.join(mvbeat_output_path, "{}.npz".format(movie_name))
    fr = np.load(mvbeat_path, allow_pickle=True)
    movie_beat = fr['beat']
    acc_list = boom_movie_beat(movie_beat, music_beat, fps)

    # accelerate
    print("--------------------accelerate--------------------------")
    result_video = change_speed(video, acc_list, acc_rate=3)

    # 融合！！
    print("--------------------融合！！--------------------------")
    audio = moviepy.editor.AudioFileClip(music_asset_path)
    length = result_video.end
    result_video = result_video.set_audio(audio)
    if(audio.end > length): # 音乐会延长
        result_video = result_video.subclip(0,length)

    result_path = './result.mp4'
    result_video.write_videofile(result_path)

if __name__ == '__main__':
    main()
